#!/usr/bin/env bash
# Start OpenSearch + GPU remote index builder (no cuVS bench).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# shellcheck disable=SC1091
if [[ -f "${ROOT}/.env" ]]; then
  set -a
  source "${ROOT}/.env"
  set +a
fi

export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-${AWS_REGION:-us-east-2}}"
export AWS_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION}}"
export REMOTE_INDEX_BUILDER_IMAGE="${REMOTE_INDEX_BUILDER_IMAGE:-opensearchproject/remote-vector-index-builder:api-latest}"
export MAX_WORKERS="${MAX_WORKERS:-1}"
export COMPOSE_PROFILES="${COMPOSE_PROFILES:-gpu}"

if [[ -n "${NGC_API_KEY:-}" ]]; then
  echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin || true
fi

if command -v sysctl >/dev/null 2>&1; then
  sysctl -w vm.max_map_count=262144 || true
fi

COMPOSE=(docker compose)
if ! docker compose version >/dev/null 2>&1; then
  if [[ -x "${ROOT}/bin/docker-compose" ]]; then
    COMPOSE=("${ROOT}/bin/docker-compose")
  else
    echo "Docker Compose v2 is required." >&2
    exit 1
  fi
fi

"${COMPOSE[@]}" --profile gpu up --build -d --wait opensearch remote-index-builder

OPENSEARCH_URL="${OPENSEARCH_URL:-http://127.0.0.1:9200}"
REMOTE_INDEX_BUILDER_URL="${REMOTE_INDEX_BUILDER_URL:-http://127.0.0.1:1025}"
export OPENSEARCH_URL REMOTE_INDEX_BUILDER_URL
# OpenSearch talks to the builder on the compose network.
export REMOTE_INDEX_BUILDER_URL_INTERNAL="${REMOTE_INDEX_BUILDER_URL_INTERNAL:-http://remote-index-builder:1025}"

echo "OpenSearch: ${OPENSEARCH_URL}"
curl -fsS "${OPENSEARCH_URL}"
echo
echo "Remote builder host port: ${REMOTE_INDEX_BUILDER_URL}"

if [[ -n "${S3_BUCKET:-}" ]]; then
  REMOTE_INDEX_BUILDER_URL="${REMOTE_INDEX_BUILDER_URL_INTERNAL}" \
    "${ROOT}/.venv/bin/python" "${ROOT}/scripts/configure_opensearch.py"
else
  echo "S3_BUCKET is not set; skip remote-build cluster configuration."
  echo "Set S3_BUCKET (and AWS credentials or an instance role) then rerun configure_opensearch.py"
fi
