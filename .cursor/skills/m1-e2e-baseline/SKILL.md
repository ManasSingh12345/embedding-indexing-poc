---
name: m1-e2e-baseline
description: >-
  Sequential Milestone-1 embed-then-index on one GPU (NIM VL + OpenSearch
  CAGRA/HNSW + cuVS Recall@10). Use when running this PoC, smoke/1M e2e,
  nsys, MIRACL, NIM, remote-index-builder, or embedding-indexing-poc.
---

# M1 e2e baseline

## Hard rules

- Run GPU jobs in the **host SSH shell**, not the Cursor agent sandbox (no `/dev/nvidia*`, no `:8000`).
- Never commit `.env`, `data/`, `vendor/`, `results/`, or credentials. `.env.example` only.
- Sequential **embed then index** on **GPU 0**. Do not start a second e2e while one is running (`pgrep -f run_e2e.py`).
- `run_e2e.py` **deletes** `miracl-en-1m-cagra` at start (`recreate()`).
- Ground truth is **cuVS brute-force cosine top-10** on the indexed vectors, not MIRACL qrels. Recall@10 is `|pred ∩ gt| / (n_queries * k)`, the same formula as cuVS `notebooks/utils.py` `calc_recall`. Target ≥ 0.95.
- Do **not** wrap NIM or the builder with `nsys` as container PID 1 (embed server never lands on the GPU).
- Keep `NIM_PERFORMANCE_MODE=1` (throughput batch 64). Latency mode roughly halved 20k embed inputs/s and used far less VRAM.
- Keep `REMOTE_BUILD_POLL_INTERVAL=200ms` and `--flush-every 0` (one flush after all bulks). The OpenSearch default poll (5s) idles ~15s per flush.
- Keep `--flush-parallel 1` and `MAX_WORKERS=1` on one GPU. Parallel CAGRA jobs on that GPU were slower (~26s each vs ~1s serial).
- Bulk defaults: `orjson`, `--bulk-docs 1000`, `--bulk-workers 4`.
- If `AWS_ENDPOINT_URL` or `S3_ENDPOINT` is set, the stack uses LocalStack (`local-s3` profile). Do not commit those endpoints with real keys.

## Commands

```bash
# Setup (once)
cp .env.example .env   # NGC_API_KEY, S3_BUCKET, AWS keys
./scripts/bootstrap.sh && ./scripts/ensure_gpu.sh && ./scripts/check_setup.sh

# Smoke / 1M
./scripts/run_e2e.sh --limit 20000
./scripts/run_e2e.sh

# Nsys (client NVTX + cuVS CUDA; GPU counters often denied on Blackwell)
./scripts/nsys_trace_all.sh
```

Unset `HTTP_PROXY`/`HTTPS_PROXY` on the host runner (sandbox proxy breaks localhost).

cuVS: `pip install --extra-index-url https://pypi.nvidia.com 'cuvs-cu13==26.8.1' 'cupy-cuda13x' orjson`. Set `CUPY_CACHE_DIR` / `CUDA_CACHE_PATH` under the repo `.cache/`.

## Nsys

- NVTX colors must be **ints** or nvtx built-in names (`blue`, `green`, …). Do not use `gray` (needs matplotlib).
- `--gpu-metrics-devices=all` → `ERR_NVGPUCTRPERM` on this RTX PRO 6000 unless `NVreg_RestrictProfilingToAdminUsers=0`. Fall back without it.
- `curl` health waits must use `--connect-timeout` / `--max-time` or they hang.
- `nsys_trace_all.sh` skips NIM restart if `:8000` is already ready.

## Outputs

- E2e: `results/m1_sequential/metrics.json`
- Nsys: `results/nsys_all/e2e.nsys-rep` (NVTX phases + host cuVS kernels only)

OpenSearch JVM is CPU-only. Remote CAGRA is `remote-index-builder` on GPU 0. Run:ai 10:1 is later, not this baseline.
