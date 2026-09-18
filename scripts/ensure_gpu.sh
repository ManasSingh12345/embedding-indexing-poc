#!/usr/bin/env bash
# Recreate the CAGRA builder with an explicit GPU, keep NIM on the same GPU,
# and print nvidia-smi / DeviceRequests for both.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

GPU_ID="${NVIDIA_VISIBLE_DEVICES:-0}"
export NVIDIA_VISIBLE_DEVICES="${GPU_ID}"
NIM_NAME="${CONTAINER_NAME:-llama-nemotron-embed-vl-1b-v2}"

COMPOSE=(docker compose)
if ! docker compose version >/dev/null 2>&1; then
  COMPOSE=("${ROOT}/bin/docker-compose")
fi

echo "==> Recreating remote-index-builder on GPU ${GPU_ID}"
"${COMPOSE[@]}" --profile gpu up -d --force-recreate --no-deps --wait remote-index-builder

echo
echo "==> Docker GPU device requests"
for name in remote-index-builder "${NIM_NAME}"; do
  echo "--- ${name} ---"
  docker inspect "${name}" --format '{{json .HostConfig.DeviceRequests}}' 2>/dev/null || echo "container not found"
  echo
done

echo "==> nvidia-smi inside containers"
for name in remote-index-builder "${NIM_NAME}"; do
  echo "--- ${name} ---"
  docker exec "${name}" nvidia-smi -L 2>/dev/null || \
    docker exec "${name}" python3 -c 'import ctypes; ctypes.CDLL("libcuda.so.1"); print("libcuda loaded")' 2>/dev/null || \
    echo "no nvidia-smi in image; check DeviceRequests above"
  echo
done

echo "==> host GPU processes"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || nvidia-smi

echo
echo "OpenSearch JVM stays on CPU. GPU indexing is remote-index-builder (CAGRA)."
echo "NIM and the builder now share GPU ${GPU_ID}. Sequential embed-then-index is expected."
echo "Re-check cluster builder endpoint:"
curl -fsS http://127.0.0.1:9200/_cluster/settings | python3 -m json.tool | head -30
