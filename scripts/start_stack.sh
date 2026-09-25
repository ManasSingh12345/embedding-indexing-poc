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
if [[ -n "${S3_ENDPOINT:-}" || -n "${AWS_ENDPOINT_URL:-}" ]]; then
  case ",${COMPOSE_PROFILES}," in
    *,local-s3,*) ;;
    *) export COMPOSE_PROFILES="${COMPOSE_PROFILES},local-s3" ;;
  esac
fi

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

UP_SERVICES=(opensearch remote-index-builder)
if [[ ",${COMPOSE_PROFILES}," == *",local-s3,"* ]]; then
  UP_SERVICES+=(localstack)
fi
"${COMPOSE[@]}" --profile gpu --profile local-s3 up --build -d --wait "${UP_SERVICES[@]}"

if [[ -n "${S3_BUCKET:-}" && -n "${S3_ENDPOINT:-}${AWS_ENDPOINT_URL:-}" ]]; then
  echo "==> Ensuring S3 bucket ${S3_BUCKET} exists"
  "${ROOT}/.venv/bin/python" - <<'PY'
import os
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

endpoint = (os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("S3_ENDPOINT") or "").strip()
# Host-side client cannot resolve compose DNS names.
endpoint = endpoint.replace("http://minio:", "http://127.0.0.1:").replace("https://minio:", "https://127.0.0.1:")
endpoint = endpoint.replace("http://localstack:", "http://127.0.0.1:").replace("https://localstack:", "https://127.0.0.1:")
bucket = os.environ["S3_BUCKET"]
region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION") or "us-east-1"
kwargs = {"region_name": region, "config": Config(s3={"addressing_style": "path"})}
if endpoint:
    kwargs["endpoint_url"] = endpoint
s3 = boto3.client("s3", **kwargs)
try:
    s3.head_bucket(Bucket=bucket)
    print(f"bucket exists: {bucket}")
except ClientError:
    create = {"Bucket": bucket}
    if region and region != "us-east-1":
        create["CreateBucketConfiguration"] = {"LocationConstraint": region}
    s3.create_bucket(**create)
    print(f"created bucket: {bucket}")
PY
fi

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
