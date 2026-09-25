#!/usr/bin/env bash
# Capture a short Nsight Systems GPU-metrics slice of whatever is already on GPU 0.
# Safe to run during the 1M e2e: it does not launch run_e2e.py or drop the index.
#
# This records device-wide SM/HBM metrics (NIM + CAGRA builder). It does not inject
# CUDA API tracing into those containers.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

NSYS="${NSYS:-}"
if [[ -z "${NSYS}" ]]; then
  if [[ -x "${ROOT}/vendor/nsight-systems/target-linux-x64/nsys" ]]; then
    NSYS="${ROOT}/vendor/nsight-systems/target-linux-x64/nsys"
  elif command -v nsys >/dev/null 2>&1; then
    NSYS="$(command -v nsys)"
  else
    echo "nsys not found. Install the CLI under vendor/nsight-systems or set NSYS=." >&2
    exit 1
  fi
fi

DURATION="${DURATION:-90}"
OUT_DIR="${ROOT}/results/nsys_smoke"
mkdir -p "${OUT_DIR}"
REPORT="${OUT_DIR}/live_gpu_slice"
SESSION="e2e_live_slice"

echo "==> nsys live GPU-metrics slice (${DURATION}s)"
"${NSYS}" --version
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv

# Drop a leftover session if a previous slice was interrupted.
"${NSYS}" sessions list 2>/dev/null || true
"${NSYS}" cancel --session="${SESSION}" 2>/dev/null || true

"${NSYS}" start \
  --session-new="${SESSION}" \
  --gpu-metrics-devices=all \
  --sample=none \
  --cpuctxsw=none \
  --output="${REPORT}" \
  --force-overwrite=true

echo "==> collecting ${DURATION}s (Ctrl-C then: ${NSYS} stop --session=${SESSION})"
sleep "${DURATION}"
"${NSYS}" stop --session="${SESSION}"

echo "==> ${REPORT}.nsys-rep"
ls -lh "${REPORT}".* 2>/dev/null || true
"${NSYS}" stats "${REPORT}.nsys-rep" | tee "${OUT_DIR}/live_gpu_slice_stats.txt" || true
