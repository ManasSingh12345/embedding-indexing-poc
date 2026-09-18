#!/usr/bin/env bash
# Pull llama-nemotron-embed-vl-1b-v2 NIM and pre-fetch model weights.
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

mkdir -p "${LOCAL_NIM_CACHE}/cache" "${LOCAL_NIM_CACHE}/weights"

echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin

echo "Pulling ${IMG_NAME}"
docker pull "${IMG_NAME}"

echo "Pre-fetching model weights (no GPU required for this step)"
docker run --rm --name="${CONTAINER_NAME}-download" \
  -e NIM_ENGINE_MODEL_DOWNLOAD_ONLY=1 \
  -e NIM_ENGINE_MODEL_DOWNLOAD_PROVIDER=ngc \
  -e NGC_API_KEY \
  -v "${LOCAL_NIM_CACHE}/weights:/model" \
  -u "$(id -u)" \
  "${IMG_NAME}"

echo "NIM image and weights are ready under ${LOCAL_NIM_CACHE}"
echo "Start the server with: ${ROOT}/scripts/start_nim.sh"
