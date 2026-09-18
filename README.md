# Embedding & indexing PoC (Milestone 1)

Sequential **embed → GPU CAGRA/HNSW index → search** on **one** NVIDIA RTX PRO 6000. Measures e2e throughput, latency, and **Recall@10 vs cuVS brute-force** (not MIRACL qrels).

```
MIRACL texts → NIM (GPU 0) → OpenSearch bulk
                 flush → remote-index-builder CAGRA (same GPU)
                 → cuVS GT top-10 → OpenSearch kNN → metrics.json
```

NIM and the builder **share GPU 0**. They run one after the other in each 10k microbatch. OpenSearch JVM stays on CPU. Run:ai 10:1 sharing is out of scope for this baseline.

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

## Nsight Systems

Do **not** wrap the NIM container with `nsys` as PID 1 (GPU goes idle). Profile the Python client; GPU *metrics* need `NVreg_RestrictProfilingToAdminUsers=0` on this Blackwell (otherwise `ERR_NVGPUCTRPERM`).

```bash
# After NIM is healthy (~16 GiB). Skips GPU counters if denied.
./scripts/nsys_trace_all.sh
```

NVTX ranges: `embed_batch`, `index_batch`, `cuvs_bruteforce_topk10`, `opensearch_search`. Report: `results/nsys_all/e2e.nsys-rep`. That file has **cuVS kernels + client NVTX**, not NIM kernel names.

## Agent notes

Project skill: [`.cursor/skills/m1-e2e-baseline/SKILL.md`](.cursor/skills/m1-e2e-baseline/SKILL.md). OpenSearch GPU builder details: [`deploy/DEPLOYMENT.md`](deploy/DEPLOYMENT.md).
