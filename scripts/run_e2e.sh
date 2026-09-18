#!/usr/bin/env bash
# Host runner: sequential 1M embed → CAGRA index → cuVS brute-force Recall@10.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

# Do not inherit the Cursor sandbox proxy on a normal host shell.
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy || true

PY="${ROOT}/.venv/bin/python"
PIP="${ROOT}/.venv/bin/pip"
export CUPY_CACHE_DIR="${ROOT}/.cache/cupy"
export CUDA_CACHE_PATH="${ROOT}/.cache/nv"
mkdir -p "${CUPY_CACHE_DIR}" "${CUDA_CACHE_PATH}" "${ROOT}/results/m1_sequential"

echo "==> Checking NIM / OpenSearch / builder"
curl -fsS "http://127.0.0.1:${NIM_PORT:-8000}/v1/health/ready"
echo
curl -fsS http://127.0.0.1:9200/_cluster/health
echo
python3 -c 'import socket; socket.create_connection(("127.0.0.1",1025),3).close(); print("builder ok")'

echo "==> Installing cuVS + CuPy if needed"
"${PIP}" install --extra-index-url https://pypi.nvidia.com 'cuvs-cu13==26.8.1' 'cupy-cuda13x'

echo "==> GPU"
nvidia-smi -L
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv

echo "==> Running e2e"
mkdir -p "${ROOT}/results/m1_sequential"
exec "${PY}" -u "${ROOT}/scripts/run_e2e.py" "$@"
