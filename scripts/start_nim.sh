#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
if [[ -f "${ROOT}/.env" ]]; then
  set -a
  source "${ROOT}/.env"
  set +a
fi

: "${NGC_API_KEY:?Set NGC_API_KEY in the environment or ${ROOT}/.env}"

export NIM_MODEL_NAME="${NIM_MODEL_NAME:-nvidia/llama-nemotron-embed-vl-1b-v2}"
export CONTAINER_NAME="${CONTAINER_NAME:-$(basename "${NIM_MODEL_NAME}")}"
export IMG_NAME="${IMG_NAME:-nvcr.io/nim/nvidia/${CONTAINER_NAME}:2.3.0}"
export LOCAL_NIM_CACHE="${LOCAL_NIM_CACHE:-${ROOT}/.cache/nim}"
export NIM_PORT="${NIM_PORT:-8000}"

mkdir -p "${LOCAL_NIM_CACHE}/cache" "${LOCAL_NIM_CACHE}/weights"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

# Same GPU as the CAGRA builder (sequential embed-then-index on one Pro 6000).
GPU_ID="${NVIDIA_VISIBLE_DEVICES:-0}"

docker run -d --name="${CONTAINER_NAME}" \
  --runtime=nvidia \
  --gpus "device=${GPU_ID}" \
  --shm-size=16GB \
  -e NVIDIA_VISIBLE_DEVICES="${GPU_ID}" \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e NIM_ENGINE_MODEL_DOWNLOAD_PROVIDER=ngc \
  -e NGC_API_KEY \
  -v "${LOCAL_NIM_CACHE}/cache:/opt/cache" \
  -v "${LOCAL_NIM_CACHE}/weights:/model" \
  -u "$(id -u)" \
  -p "${NIM_PORT}:8000" \
  "${IMG_NAME}"

echo "Waiting for NIM at http://127.0.0.1:${NIM_PORT}/v1/health/ready"
for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${NIM_PORT}/v1/health/ready" >/dev/null 2>&1; then
    curl -fsS "http://127.0.0.1:${NIM_PORT}/v1/health/ready"
    echo
    exit 0
  fi
  sleep 5
done

echo "NIM did not become ready in time. Logs:" >&2
docker logs --tail 80 "${CONTAINER_NAME}" >&2
exit 1
