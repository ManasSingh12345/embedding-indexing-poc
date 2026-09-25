# Embedding & indexing PoC (Milestone 1)

Sequential **embed → GPU CAGRA/HNSW index → search** on **one** GPU (measured on an H100 NVL). Measures e2e throughput, latency, and **Recall@10 vs cuVS brute-force** (not MIRACL qrels).

```
MIRACL texts → NIM (GPU 0) → OpenSearch bulk
                 flush → remote-index-builder CAGRA (same GPU)
                 → cuVS GT top-10 → OpenSearch kNN → metrics.json
```

NIM and the builder **share GPU 0**. Each microbatch embeds, then bulks. The remote CAGRA build runs after the last bulk, one graph per Lucene segment. OpenSearch JVM stays on CPU. Run:ai 10:1 sharing is out of scope for this baseline.

## Not in this repo

| Keep local | Why |
|---|---|
| `.env` | NGC + AWS keys |
| `data/` | MIRACL 1M |
| `vendor/` | Nsight CLI, cloned cuVS / builder |
| `results/` | embeddings, nsys reports |
| `.venv/`, `.cache/` | Python + NIM weights |

Copy `.env.example` → `.env`. Never commit keys or corpus files.

## Stack

| Piece | What |
|---|---|
| NIM | `nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:2.3.0`, 1024-d FP32, L2-normalized |
| OpenSearch | 3.6.0 + kNN + S3 repo; Faiss HNSW inner product |
| Builder | `opensearchproject/remote-vector-index-builder` on GPU 0 (CAGRA → HNSW) |
| GT / recall | cuVS 26.8 brute-force cosine top-10 |

## Setup (GPU host shell)

Cursor’s agent sandbox cannot see `/dev/nvidia*` or `:8000`. Use the instance SSH/host terminal.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # set NGC_API_KEY, S3_BUCKET, AWS creds
./scripts/bootstrap.sh
./scripts/ensure_gpu.sh
./scripts/check_setup.sh
```

`bootstrap.sh` pulls NIM, downloads MIRACL 1M, starts OpenSearch + builder + NIM.

## Run

```bash
# Smoke (2 × 10k)
./scripts/run_e2e.sh --limit 20000

# Full 1M (recreates index miracl-en-1m-cagra)
./scripts/run_e2e.sh
```

Each run **deletes and recreates** the OpenSearch index. Optional S3 staging cleanup:

```bash
set -a && source .env && set +a
curl -X DELETE http://127.0.0.1:9200/miracl-en-1m-cagra
aws s3 rm "s3://${S3_BUCKET}/${S3_PREFIX:-knn-indexes}/" --recursive --region "${AWS_DEFAULT_REGION:-us-east-2}"
```

Write-up: `results/m1_sequential/metrics.json` (wall, embed vs index, vec/s, Recall@10, p50/p95, QPS). Target Recall@10 ≥ 0.95.

Unset `HTTP_PROXY` / `HTTPS_PROXY` before `run_e2e.sh`. A sandbox proxy breaks localhost calls to NIM and OpenSearch.

## Timers

| Field | What it includes |
|---|---|
| `embed_s` | NIM `/v1/embeddings` (`input_type=passage`), L2 normalize, memmap write |
| `bulk_s` | Client NDJSON plus OpenSearch `_bulk` ingest. The graph is not built here. |
| `flush_s` | `_flush` (segment commit, S3 upload, remote CAGRA → HNSW, `.faiss` download) and `_refresh`. One call still builds one graph per Lucene segment, serially. |

`index_s` is `bulk_s + flush_s`.

## Defaults

Set in `.env.example` and applied by the scripts:

| Knob | Default | Why |
|---|---|---|
| `NIM_PERFORMANCE_MODE` | `1` | NIM 2.3 throughput defaults, including pipeline batch 64. Latency mode left a 20k embed near 326 inputs/s; throughput mode reached about 709 inputs/s on an H100 NVL and used ~39 GiB instead of ~6 GiB. |
| `REMOTE_BUILD_POLL_INTERVAL` | `200ms` | OpenSearch waits `3 × poll.interval` before the first status check. The 5s default is about 15s of idle time per flush. |
| `--flush-every` | `0` | One remote build after all bulks. Flushing every 10k docs repeated that wait. |
| `--bulk-docs` / `--bulk-workers` | `1000` / `4` | Larger `orjson` NDJSON batches and parallel `_bulk` POSTs. 20k bulk fell from ~20s to ~5s. Four index threads commit up to four Lucene segments (`refresh_interval=-1` does not merge them). A 20k flush built three graphs (6,574 / 8,103 / 5,323 docs) one after another. |
| `--flush-parallel` / `MAX_WORKERS` | `1` / `1` | One index, one CAGRA job. Four concurrent builds on the same GPU stretched each ~1s graph to ~26s. |

Published H100 FP16 passage throughput (batch 64, concurrency 1, 300 tokens) is 880 inputs/s. This client's MIRACL passages are ~125 tokens and the embed timer includes HTTP and the memmap write, so 709 inputs/s is still short of that table.

## Recall

Ground truth is **cuVS brute-force cosine top-10** over the same L2-normalized vectors that were indexed, not MIRACL qrels. The 5,000 queries are document rows drawn with seed 42. OpenSearch search is inner product, `ef_search=256`, `k=10`.

Recall@10 matches cuVS [`calc_recall`](https://github.com/NVIDIA/cuvs/blob/main/notebooks/utils.py): for each query, `|set(pred[:k]) ∩ set(gt[:k])|`, divided by `n_queries * k`.

## Local S3

If `AWS_ENDPOINT_URL` or `S3_ENDPOINT` is set, `start_stack.sh` starts LocalStack and creates the bucket from the host (compose DNS names are rewritten to `127.0.0.1`). Leave those unset to use real AWS.

## Nsight Systems

Do **not** wrap NIM or the builder with `nsys` as container PID 1. Profile the Python client. GPU performance counters need `NVreg_RestrictProfilingToAdminUsers=0` (otherwise `ERR_NVGPUCTRPERM` on this host). The script uses `vendor/nsight-systems` when present, otherwise `nsys` on `PATH` (`/usr/local/bin/nsys` here).

```bash
# After NIM is healthy. Skips GPU counters if denied.
./scripts/nsys_trace_all.sh
```

Client NVTX ranges: `load_docs`, `opensearch_recreate`, `embed_batch`, `index_bulk`, `index_flush`, `cuvs_bruteforce_topk10`, `opensearch_search`. Report: `results/nsys_all/e2e.nsys-rep`. That file has **client NVTX and host cuVS kernels**. NIM and the builder are other containers, so their CUDA is not in it. System-wide CUDA injection does not cross into Docker here.

A builder capture has to start nsys inside the container (`docker run --init`) and stop it with `docker exec -u appuser nsys stop --session=...` (uid 1001; a host `kill` is EPERM). Open that `.nsys-rep` next to the client report in the Nsight GUI. There is no CLI merge. NIM export crashes in `TimeConversion.cpp` during GPU tick conversion, so there is no NIM kernel report.

On a 20k run the builder timeline is three bursts about 3s apart: one CAGRA build per Lucene segment, serial because `MAX_WORKERS=1`. Each burst is mostly k-means. The only NVTX ranges are CUB (`cub::DeviceHistogram::MultiHistogramEven`, `cub::DeviceFor::Bulk`). The graph itself is a short unannotated tail: two `compute_similarity` GEMMs, `kern_prune`, `kern_make_rev_graph`. About 166k `cudaFree`/`cudaMalloc` pairs are temporary-buffer churn (~7 per k-means step, median ~3µs), not HBM traffic. Kernel GPU time was 1.06s across 268k kernels (average 4µs); memcpy was about 4 GB/s. That loop is launch-and-allocation bound. The six long similarity GEMMs (0.24s) are the only kernels that might be compute- or bandwidth-bound, and this capture has no DRAM counters.

## Agent notes

Project skill: [`.cursor/skills/m1-e2e-baseline/SKILL.md`](.cursor/skills/m1-e2e-baseline/SKILL.md). OpenSearch GPU builder details: [`deploy/DEPLOYMENT.md`](deploy/DEPLOYMENT.md).
