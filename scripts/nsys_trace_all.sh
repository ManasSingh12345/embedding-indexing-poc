#!/usr/bin/env bash
# 20k nsys capture of the sequential e2e client, with GPU metrics for NIM/CAGRA.
#
# Container-wrapping NIM with nsys as PID 1 left the GPU empty and hung the
# health wait. Default is: restore unwrapped NIM + builder, then nsys the client.
# NIM kernel *names* still need a working in-container wrap (WRAP_CONTAINERS=1).
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
    echo "nsys not found. Install the CLI under vendor/nsight-systems or set NSYS=." >&2
    exit 1
  fi
fi

if pgrep -f "${ROOT}/scripts/run_e2e.py" >/dev/null 2>&1; then
  echo "run_e2e.py is already running." >&2
  pgrep -af 'run_e2e.py' || true
  exit 1
fi

: "${NGC_API_KEY:?Set NGC_API_KEY in ${ROOT}/.env}"

PY="${ROOT}/.venv/bin/python"
PIP="${ROOT}/.venv/bin/pip"
export CUPY_CACHE_DIR="${ROOT}/.cache/cupy"
export CUDA_CACHE_PATH="${ROOT}/.cache/nv"
OUT_DIR="${ROOT}/results/nsys_all"
mkdir -p "${CUPY_CACHE_DIR}" "${CUDA_CACHE_PATH}" "${OUT_DIR}"

LIMIT="${LIMIT:-20000}"
QUERIES="${QUERIES:-200}"
MICROBATCH="${MICROBATCH:-10000}"
NIM_PORT="${NIM_PORT:-8000}"
CONTAINER_NAME="${CONTAINER_NAME:-$(basename "${NIM_MODEL_NAME:-nvidia/llama-nemotron-embed-vl-1b-v2}")}"

COMPOSE=(docker compose)
if ! docker compose version >/dev/null 2>&1; then
  COMPOSE=("${ROOT}/bin/docker-compose")
fi

wait_http() {
  local url="$1" tries="${2:-90}"
  local i
  for i in $(seq 1 "${tries}"); do
    if curl -fsS --connect-timeout 2 --max-time 3 "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

if curl -fsS --connect-timeout 2 --max-time 3 "http://127.0.0.1:${NIM_PORT}/v1/health/ready" >/dev/null 2>&1 \
  && python3 -c 'import socket; socket.create_connection(("127.0.0.1",1025),2).close()' 2>/dev/null; then
  echo "==> NIM + builder already healthy; skipping restart"
else
  echo "==> Restoring unwrapped NIM + builder"
  if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    docker rm -f "${CONTAINER_NAME}" >/dev/null || true
  fi
  "${ROOT}/scripts/start_nim.sh"
  "${COMPOSE[@]}" --profile gpu up -d --no-deps --force-recreate --wait remote-index-builder
fi

echo "==> Health"
if ! wait_http "http://127.0.0.1:${NIM_PORT}/v1/health/ready" 90; then
  echo "NIM not ready. Logs:" >&2
  docker logs --tail 80 "${CONTAINER_NAME}" >&2
  exit 1
fi
curl -fsS --max-time 5 "http://127.0.0.1:${NIM_PORT}/v1/health/ready"; echo
curl -fsS --max-time 5 http://127.0.0.1:9200/_cluster/health; echo
python3 -c 'import socket; socket.create_connection(("127.0.0.1",1025),5).close(); print("builder ok")'
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv

echo "==> nvtx + cuVS"
"${PIP}" install -q nvtx
"${PIP}" install -q --extra-index-url https://pypi.nvidia.com 'cuvs-cu13==26.8.1' 'cupy-cuda13x'

echo "==> nsys profile 20k e2e (CUDA + NVTX; GPU metrics if permitted)"
PROBE="${OUT_DIR}/.nsys_probe"
GPU_METRICS=""
if "${NSYS}" profile \
    --force-overwrite=true \
    --output="${PROBE}" \
    --duration=2 \
    --sample=none \
    --trace=nvtx \
    --gpu-metrics-devices=all \
    --stats=false \
    /bin/sleep 2; then
  GPU_METRICS="--gpu-metrics-devices=all"
  echo "GPU metrics: on"
else
  echo "WARN: GPU performance counters denied (ERR_NVGPUCTRPERM)." >&2
  echo "      Continuing without --gpu-metrics-devices. NVTX/CUDA still collected." >&2
  echo "      To enable metrics later (needs reboot/reload):" >&2
  echo "        sudo nvidia-smi -pm 1" >&2
  echo "        echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nsight.conf" >&2
fi

# shellcheck disable=SC2086
"${NSYS}" profile \
  --force-overwrite=true \
  --output="${OUT_DIR}/e2e" \
  --trace=cuda,nvtx,osrt \
  --sample=process-tree \
  --backtrace=fp \
  --python-sampling=true \
  --python-sampling-frequency=1000 \
  --stats=true \
  ${GPU_METRICS} \
  "${PY}" -u "${ROOT}/scripts/run_e2e.py" \
    --limit "${LIMIT}" \
    --queries "${QUERIES}" \
    --microbatch "${MICROBATCH}" \
    --out-dir "${OUT_DIR}"

echo "==> reports"
ls -lh "${OUT_DIR}/e2e".* "${OUT_DIR}/metrics.json" 2>/dev/null || true
"${NSYS}" stats \
  --report nvtx_sum \
  --report cuda_gpu_kern_sum \
  --format table \
  "${OUT_DIR}/e2e.nsys-rep" \
  | tee "${OUT_DIR}/e2e_stats.txt" \
  || true
echo "Open ${OUT_DIR}/e2e.nsys-rep in Nsight Systems."
echo "NVTX ranges: load_docs, opensearch_recreate, embed_batch, index_bulk, index_flush, cuvs_bruteforce_topk10, opensearch_search."
if [[ -z "${GPU_METRICS}" ]]; then
  echo "GPU SM-counter metrics were skipped (ERR_NVGPUCTRPERM). Kernel names from this process are still in cuda_gpu_kern_sum (cuVS only)."
fi
