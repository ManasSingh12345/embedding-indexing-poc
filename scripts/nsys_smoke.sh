#!/usr/bin/env bash
# Profile the sequential e2e client with Nsight Systems on a 20k smoke.
#
# What this captures
#   - This Python process: NVTX ranges (embed_batch / index_batch / cuVS / search),
#     CUDA from host cuVS, OS runtime, CPU samples, Python samples.
#   - Whole-GPU metrics (SM / HBM) for whoever is on GPU 0, including NIM and the
#     remote CAGRA builder — even though those are sibling Docker processes.
#
# What this does NOT capture
#   - CUDA API / kernel traces inside the NIM and remote-index-builder containers.
#     nsys only traces CUDA in the launched process tree. GPU *metrics* still show
#     their utilization. To get their kernel traces, wrap the container entrypoint
#     with nsys, or run `scripts/nsys_live_slice.sh` during a live job (metrics only).
#
# Run on the host shell that can see :8000 / :9200 (not the Cursor agent sandbox):
#   /home/ec2-user/embedding-indexing-poc/scripts/nsys_smoke.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy || true

NSYS="${NSYS:-}"
if [[ -z "${NSYS}" ]]; then
  if [[ -x "${ROOT}/vendor/nsight-systems/target-linux-x64/nsys" ]]; then
    NSYS="${ROOT}/vendor/nsight-systems/target-linux-x64/nsys"
  elif [[ -x "${ROOT}/vendor/nsight-systems/host-linux-x64/nsys" ]]; then
    NSYS="${ROOT}/vendor/nsight-systems/host-linux-x64/nsys"
  elif command -v nsys >/dev/null 2>&1; then
    NSYS="$(command -v nsys)"
  else
    echo "nsys not found. Install the CLI tarball under vendor/nsight-systems or set NSYS=." >&2
    exit 1
  fi
fi

if pgrep -f "${ROOT}/scripts/run_e2e.py" >/dev/null 2>&1; then
  echo "run_e2e.py is already running. This smoke recreates the OpenSearch index;" >&2
  echo "wait for the live job to finish (or Ctrl-C it) before profiling a smoke." >&2
  pgrep -af 'run_e2e.py' || true
  exit 1
fi

PY="${ROOT}/.venv/bin/python"
PIP="${ROOT}/.venv/bin/pip"
export CUPY_CACHE_DIR="${ROOT}/.cache/cupy"
export CUDA_CACHE_PATH="${ROOT}/.cache/nv"
OUT_DIR="${ROOT}/results/nsys_smoke"
mkdir -p "${CUPY_CACHE_DIR}" "${CUDA_CACHE_PATH}" "${OUT_DIR}"
REPORT="${OUT_DIR}/e2e_smoke"
PROBE="${OUT_DIR}/.nsys_probe"

LIMIT="${LIMIT:-20000}"
QUERIES="${QUERIES:-200}"
MICROBATCH="${MICROBATCH:-10000}"

echo "==> nsys: ${NSYS}"
"${NSYS}" --version

echo "==> Checking NIM / OpenSearch / builder"
curl -fsS "http://127.0.0.1:${NIM_PORT:-8000}/v1/health/ready"
echo
curl -fsS http://127.0.0.1:9200/_cluster/health
echo
python3 -c 'import socket; socket.create_connection(("127.0.0.1",1025),3).close(); print("builder ok")'

echo "==> Ensuring nvtx + cuVS"
"${PIP}" install -q nvtx
"${PIP}" install -q --extra-index-url https://pypi.nvidia.com 'cuvs-cu13==26.8.1' 'cupy-cuda13x'

echo "==> GPU"
nvidia-smi -L
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv

GPU_METRICS=""
echo "==> Probing GPU metrics"
if "${NSYS}" profile \
    --force-overwrite=true \
    --output="${PROBE}" \
    --duration=3 \
    --sample=none \
    --trace=nvtx \
    --gpu-metrics-devices=all \
    --stats=false \
    /bin/sleep 3; then
  GPU_METRICS="--gpu-metrics-devices=all"
  echo "GPU metrics: on"
else
  echo "WARN: --gpu-metrics-devices=all not available; continuing without it" >&2
fi

run_profile() {
  local extra="$1"
  # shellcheck disable=SC2086
  "${NSYS}" profile \
    --force-overwrite=true \
    --output="${REPORT}" \
    --trace=cuda,nvtx,osrt \
    --sample=process-tree \
    --backtrace=fp \
    --python-sampling=true \
    --python-sampling-frequency=1000 \
    --stats=true \
    ${extra} \
    "${PY}" -u "${ROOT}/scripts/run_e2e.py" \
      --limit "${LIMIT}" \
      --queries "${QUERIES}" \
      --microbatch "${MICROBATCH}" \
      --out-dir "${OUT_DIR}"
}

echo "==> nsys profile smoke (limit=${LIMIT} queries=${QUERIES})"
if ! run_profile "${GPU_METRICS}"; then
  echo "WARN: nsys failed with python-sampling/GPU extras; retrying CUDA+NVTX only" >&2
  "${NSYS}" profile \
    --force-overwrite=true \
    --output="${REPORT}" \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --stats=true \
    "${PY}" -u "${ROOT}/scripts/run_e2e.py" \
      --limit "${LIMIT}" \
      --queries "${QUERIES}" \
      --microbatch "${MICROBATCH}" \
      --out-dir "${OUT_DIR}"
fi

echo "==> nsys stats (NVTX + CUDA kernels)"
"${NSYS}" stats \
  --report nvtx_sum \
  --report cuda_gpu_kern_sum \
  --format table \
  "${REPORT}.nsys-rep" \
  | tee "${OUT_DIR}/nsys_stats.txt" \
  || true

echo "==> reports"
ls -lh "${REPORT}".* "${OUT_DIR}/metrics.json" 2>/dev/null || true
echo "Open ${REPORT}.nsys-rep in the Nsight Systems GUI."
echo "CLI: ${NSYS} stats ${REPORT}.nsys-rep"
