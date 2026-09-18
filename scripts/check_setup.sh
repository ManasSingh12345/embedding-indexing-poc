#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "project: ${ROOT}"
echo "python:  ${ROOT}/.venv/bin/python ($("${ROOT}/.venv/bin/python" -V 2>/dev/null || echo missing))"
echo "compose: $("${ROOT}/bin/docker-compose" version 2>/dev/null || echo missing)"
echo "crane:   $("${ROOT}/bin/crane" version 2>/dev/null || echo missing)"
echo
echo "--- GPU ---"
nvidia-smi -L 2>/dev/null || echo "nvidia-smi not available in this process"
echo
echo "--- Docker ---"
docker info >/dev/null 2>&1 && echo "docker daemon: ok" || echo "docker daemon: not reachable"
echo
echo "--- Files ---"
ls -ld "${ROOT}/deploy/opensearch" "${ROOT}/vendor/cuvs/deploy" "${ROOT}/vendor/remote-vector-index-builder" 2>/dev/null
if [[ -f "${ROOT}/data/miracl-en-1m/manifest.json" ]]; then
  echo "MIRACL:"
  cat "${ROOT}/data/miracl-en-1m/manifest.json"
else
  echo "MIRACL: not downloaded yet"
fi
if [[ -d "${ROOT}/.cache/nim/weights" ]]; then
  echo "NIM weights cache:"
  du -sh "${ROOT}/.cache/nim/weights" 2>/dev/null || true
fi
echo
echo "--- Services ---"
curl -fsS --max-time 3 http://127.0.0.1:8000/v1/health/ready && echo || echo "NIM not listening on :8000"
curl -fsS --max-time 3 http://127.0.0.1:9200 && echo || echo "OpenSearch not listening on :9200"
python3 -c 'import socket; socket.create_connection(("127.0.0.1",1025),2).close(); print("builder :1025 reachable")' 2>/dev/null || echo "remote index builder not listening on :1025"
