#!/usr/bin/env bash
# Restart NIM and the remote index builder with the memory split in config/pipeline.yaml.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy || true
"${ROOT}/scripts/start_stack.sh"
"${ROOT}/scripts/start_nim.sh"
