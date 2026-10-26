---
name: m1-e2e-baseline
description: >-
  Single-GPU MIRACL embed + OpenSearch CAGRA/HNSW with Run:ai GPU fractions
  from config/pipeline.yaml. Hard caps need driver r615+. Use when running
  this PoC on a new machine, changing the embed:index ratio, smoke/1M e2e,
  nsys, NIM, remote-index-builder, or embedding-indexing-poc.
---

# M1 e2e baseline

## Hard rules

- Run GPU jobs in the **host SSH shell**, not the Cursor agent sandbox (no `/dev/nvidia*`, no `:8000`).
- Never commit `.env`, `data/`, `vendor/`, `results/`, `.cache/`, `images/`, or credentials. `.env.example` only.
- Do not start a second e2e while one is running (`pgrep -f scripts/run_e2e.py`; ignore a hit that is only the shell line you just typed). `run_e2e.py` deletes `miracl-en-1m-cagra` at start.
- The run **overlaps** the next embed with the previous remote flush. Only one CAGRA build is in flight. `embed_s` is the sum of embed calls. `embed_index_wall_s` is the overlapped schedule. `--flush-every 0` is one build after all bulks.
- Ground truth is **cuVS brute-force cosine top-10**, not MIRACL qrels. Recall@10 is `|pred ∩ gt| / (n_queries * k)`. Target ≥ 0.95.
- Do **not** wrap NIM or the builder with `nsys` as container PID 1. NIM export dies in `TimeConversion.cpp`. There is no usable NIM `.nsys-rep`.
- GPU split is `config/pipeline.yaml` (`embed_index_ratio` `10:1`, `dynamic_embed: true`). Memory is capped. Cores are not partitioned; both processes can use the SMs. Do not give both pods `nvidia.com/gpu: 1`.
- Hard caps need driver **r615+** and container toolkit **≥ 1.20.1**. `gpu_fraction_enforced` in `metrics.json` is that fact. `scripts/install-k8s-runai.sh` refuses any driver other than **615.71.10** and does not upgrade it.
- Builder image is `opensearchstaging/remote-vector-index-builder:api-snapshot` (cuVS 26.08, CUDA 12.9, `sm_120`). `api-latest` is CUDA 12.4 and fails on Blackwell with `cudaErrorInvalidValue` before any `.faiss` file. The served index is still FAISS HNSW.
- The builder process reads `AWS_DEFAULT_REGION` and `S3_ENDPOINT_URL`. Setting only `AWS_REGION` / `S3_ENDPOINT` fails every job in S3 client setup. OpenSearch then builds HNSW on CPU. For LocalStack, also set `AWS_REQUEST_CHECKSUM_CALCULATION=WHEN_REQUIRED` and `AWS_RESPONSE_CHECKSUM_VALIDATION=WHEN_REQUIRED`.
- A few-hundred-doc segment can fail CAGRA with "too many invalid or duplicated neighbor nodes" even on the snapshot image. `BULK_WORKERS=4` creates several segments per flush. Five of six 20k segments still built on GPU.
- `NIM_PERFORMANCE_MODE=1` OOMs at startup on this card when the builder also holds a memory reservation. The working value here is `0`.
- Keep `REMOTE_BUILD_POLL_INTERVAL=200ms`, `--flush-parallel 1`, and `MAX_WORKERS=1`.
- OpenSearch is CPU-only. From a container, the builder is the Docker bridge gateway on port 1025, not `127.0.0.1`. KAI admits per-container `gpu-memory.request` and `.limit` only. Do not also set `gpu-fraction` on the same pod. Requests must fit the card or the second pod stays Pending. A rolling update that keeps hostPort 8000 or 1025 deadlocks; use `Recreate`.

## Commands

```bash
cp .env.example .env
python3 scripts/pipeline_config.py
./scripts/bootstrap.sh
./scripts/run_e2e.sh --limit 20000
./scripts/run_e2e.sh --microbatch 10000 --flush-every 10000 --out-dir results/runai-1m-overlap
# one embed, then one index:
./scripts/run_e2e.sh --microbatch 1000000 --flush-every 0 --out-dir results/runai-1m-fullbatch
```

Unset `HTTP_PROXY`/`HTTPS_PROXY`. Single-node Run:ai install: `bash scripts/install-k8s-runai.sh`. Host cuVS for ground truth: `cuvs-cu13==26.8.1` and `cupy-cuda13x` from `pypi.nvidia.com`.

## Nsys

Client trace only: `./scripts/nsys_trace_all.sh` → `results/nsys_all/e2e.nsys-rep`. NIM and the builder are not in that file. `--gpu-metrics-devices=all` often returns `ERR_NVGPUCTRPERM`. NVTX colors must be ints or built-in names (`blue`, `green`). Do not use `gray`.

`BULK_WORKERS=4` commits several Lucene segments per flush, so one `_flush` starts several serial CAGRA builds (`MAX_WORKERS=1`).

## Outputs

- `results/runai/metrics.json` unless `--out-dir` is set. A new run overwrites that file.
- Log lines `overlap embed|flush N started|finished` are the schedule.

OpenSearch JVM is CPU-only. Remote CAGRA is `remote-index-builder` on the same GPU as NIM. How-to: `notebooks/embed_index_runai.ipynb`.
