#!/usr/bin/env bash
# Host bootstrap: NIM + MIRACL 1M + OpenSearch + CAGRA remote index builder.
# Run this on the GPU instance as root/ec2-user (not inside the Cursor sandbox).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
else
  echo "Copy ${ROOT}/.env.example to ${ROOT}/.env and set NGC_API_KEY (and S3_BUCKET)." >&2
  exit 1
fi

: "${NGC_API_KEY:?NGC_API_KEY is required to pull NIM and (optional) the NGC builder image}"

export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-${AWS_REGION:-us-east-2}}"
export AWS_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION}}"
export IMG_NAME="${IMG_NAME:-nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:2.3.0}"
export REMOTE_INDEX_BUILDER_IMAGE="${REMOTE_INDEX_BUILDER_IMAGE:-opensearchproject/remote-vector-index-builder:api-latest}"
export LOCAL_NIM_CACHE="${LOCAL_NIM_CACHE:-${ROOT}/.cache/nim}"
export HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"
export COMPOSE_PROFILES="${COMPOSE_PROFILES:-gpu}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is not installed." >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not running or this user cannot reach /var/run/docker.sock." >&2
  exit 1
fi
if ! nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi failed. The RTX Pro 6000 must be visible before starting NIM/builder." >&2
  exit 1
fi

mkdir -p "${ROOT}/bin" "${LOCAL_NIM_CACHE}/cache" "${LOCAL_NIM_CACHE}/weights" "${ROOT}/data"
if ! docker compose version >/dev/null 2>&1; then
  mkdir -p "${HOME}/.docker/cli-plugins"
  ln -sfn "${ROOT}/bin/docker-compose" "${HOME}/.docker/cli-plugins/docker-compose"
fi

sysctl -w vm.max_map_count=262144 || true

echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin

echo "==> Pulling NIM ${IMG_NAME}"
docker pull "${IMG_NAME}"

echo "==> Prefetching NIM weights"
docker run --rm --name=llama-nemotron-embed-vl-1b-v2-download \
  -e NIM_ENGINE_MODEL_DOWNLOAD_ONLY=1 \
  -e NIM_ENGINE_MODEL_DOWNLOAD_PROVIDER=ngc \
  -e NGC_API_KEY \
  -v "${LOCAL_NIM_CACHE}/weights:/model" \
  -u "$(id -u)" \
  "${IMG_NAME}"

echo "==> Pulling OpenSearch + remote index builder"
docker pull opensearchproject/opensearch:3.6.0 || true
docker pull "${REMOTE_INDEX_BUILDER_IMAGE}"

echo "==> Downloading MIRACL English corpus (1M docs)"
"${ROOT}/.venv/bin/python" "${ROOT}/scripts/download_miracl.py" --limit 1000000

echo "==> Starting OpenSearch + GPU remote index builder"
"${ROOT}/scripts/start_stack.sh"

echo "==> Starting NIM"
"${ROOT}/scripts/start_nim.sh"

echo
echo "Ready:"
echo "  NIM:        http://127.0.0.1:${NIM_PORT:-8000}/v1/health/ready"
echo "  OpenSearch: http://127.0.0.1:9200"
echo "  Builder:    http://127.0.0.1:1025"
echo "  MIRACL:     ${ROOT}/data/miracl-en-1m"
