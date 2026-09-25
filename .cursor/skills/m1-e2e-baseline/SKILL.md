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
- Sequential **embed then index** on **GPU 0**. Do not start a second e2e while one is running (`pgrep -f scripts/run_e2e.py`; ignore a hit that is only the shell line you just typed).
- `run_e2e.py` **deletes** `miracl-en-1m-cagra` at start (`recreate()`).
- Ground truth is **cuVS brute-force cosine top-10** on the indexed vectors, not MIRACL qrels. Recall@10 is `|pred ∩ gt| / (n_queries * k)`, the same formula as cuVS `notebooks/utils.py` `calc_recall`. Target ≥ 0.95.
- Do **not** wrap NIM or the builder with `nsys` as container PID 1 (`docker run --init`, so docker-init is PID 1). NIM's nsys parent can collect CUDA, but export dies in `TimeConversion.cpp` (`ConvertGpuTicksToSyncNs`); there is no usable NIM `.nsys-rep`. Do not retry that export path.
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

# Nsys client only (NVTX + host cuVS). Host CLI is /usr/local/bin/nsys;
# the script falls back to PATH when vendor/nsight-systems is absent.
# GPU counters are often ERR_NVGPUCTRPERM on this host.
./scripts/nsys_trace_all.sh
```

Unset `HTTP_PROXY`/`HTTPS_PROXY` on the host runner (sandbox proxy breaks localhost).

cuVS: `pip install --extra-index-url https://pypi.nvidia.com 'cuvs-cu13==26.8.1' 'cupy-cuda13x' orjson`. Set `CUPY_CACHE_DIR` / `CUDA_CACHE_PATH` under the repo `.cache/`.

## Nsys

- Client NVTX ranges: `load_docs`, `opensearch_recreate`, `embed_batch`, `index_bulk`, `index_flush`, `cuvs_bruteforce_topk10`, `opensearch_search`. Colors must be **ints** or nvtx built-in names (`blue`, `green`, …). Do not use `gray` (needs matplotlib).
- `results/nsys_all/e2e.nsys-rep` is the **client only**. NIM and the builder are sibling containers, so their CUDA is not in that file. `--cuda-trace-scope=system-wide` does not inject into Docker here.
- `--gpu-metrics-devices=all` → `ERR_NVGPUCTRPERM` on this host (4× H100 NVL) unless `NVreg_RestrictProfilingToAdminUsers=0`. Fall back without it. No DRAM throughput counters, so do not call the builder memory-bandwidth bound from this trace.
- `curl` health waits must use `--connect-timeout` / `--max-time` or they hang.
- `nsys_trace_all.sh` skips NIM restart if `:8000` is already ready.
- Builder CUDA trace: nsys as the parent inside the container (`--init`), then `docker exec -u appuser nsys stop --session=...`. A host `kill` of that pid is EPERM (uid 1001). `nsys stop` does not take `--output`; it uses the profile `--output`. Open the client and builder `.nsys-rep` files together in the GUI. There is no CLI merge. Set the same `NSYS_HW_ID` on both so Nsight treats them as one machine.
- NIM image has no `/bin/sh` and no `libutil.so.1` (glibc 2.43). Mounting host `dash` and `libutil.so.1` lets `nsys --version` run; export still crashes. Builder image (glibc 2.39) already has both.
- Container captures: `--sample=none --cpuctxsw=none`. CPU IP sampling is disabled on this host.

## What the 20k builder trace showed

One `_flush`, one shard, **three** CAGRA builds. `--bulk-workers 4` leaves one Lucene in-memory segment per indexing thread. `refresh_interval=-1` does not merge them before commit. The 20k index was segments of **6,574 / 8,103 / 5,323** docs. `MAX_WORKERS=1` builds those graphs serially: about **1.2–1.7s** of kernels, then about **1.1–1.3s** with no kernels (S3 round trip). Start-to-start is about **3s**. That is the three bursts.

Builder NVTX is only CUB: `cub::DeviceHistogram::MultiHistogramEven` and `cub::DeviceFor::Bulk`, plus a few sort ranges. That band is k-means for the IVF-PQ codebooks (normalize → small GEMM → histogram → `adjust_centers`). The CAGRA graph has no NVTX: two `compute_similarity` GEMMs (~20–50ms), then `kern_prune` and `kern_make_rev_graph`.

**166k** `cudaFree` paired with **166k** `cudaMalloc` (~7 per center update, median ~3µs). Raw CUDA allocator, not a pool. A few 20–50ms frees are `cudaFree` blocked on `compute_similarity`, not large frees.

Not HBM-bound. **268k** kernels, **1.06s** GPU time, average **4µs**. About **71%** of GPU time is kernels under 20µs. Memcpy was ~**4 GB/s**. The six `compute_similarity` kernels are **0.24s** and are the only ones long enough to be compute- or bandwidth-bound.

## Outputs

- E2e: `results/m1_sequential/metrics.json` (`hardware` is `nvidia-smi`, not a hardcoded name)
- Client nsys: `results/nsys_all/e2e.nsys-rep` (client NVTX + host cuVS kernels only)

OpenSearch JVM is CPU-only. Remote CAGRA is `remote-index-builder` on GPU 0. Run:ai 10:1 is later, not this baseline.
