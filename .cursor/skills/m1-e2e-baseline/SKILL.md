---
name: m1-e2e-baseline
description: >-
  Single-GPU MIRACL embed + OpenSearch CAGRA/HNSW with Run:ai GPU fractions
  from config/pipeline.yaml. Hard caps need driver r615+. Use when running
  this PoC on a new machine, changing the embed:index ratio, smoke/1M e2e,
  nsys, NIM, remote-index-builder, --cpu-index, or embedding-indexing-poc.
---

# M1 e2e baseline

## Hard rules

- Run GPU jobs in the **host SSH shell**, not the Cursor agent sandbox (no `/dev/nvidia*`, no `:8000`). Agents often cannot `sudo` or spawn a sandbox here; the user's terminal is the reliable path.
- Never commit `.env`, `data/`, `vendor/`, `results/`, `.cache/`, `images/`, or credentials. `.env.example` only.
- Do not start a second e2e while one is running (`pgrep -f scripts/run_e2e.py`; ignore a hit that is only the shell line you just typed). `run_e2e.py` deletes `miracl-en-1m-cagra` at start.
- Overlap is the code path: embed N+1 while flush N runs. Only one remote build is in flight. `embed_s` is the sum of embed calls. `embed_index_wall_s` is the overlapped schedule. `--flush-every 0` is one build after all bulks. `--cpu-index` builds FAISS HNSW on the OpenSearch JVM (`index_build=cpu_hnsw`); default is remote GPU CAGRA (`gpu_cagra`). The served index is still FAISS HNSW.
- Ground truth is **cuVS brute-force cosine top-10**, not MIRACL qrels. Recall@10 is `|pred ∩ gt| / (n_queries * k)`. Target ≥ 0.95.
- Do **not** wrap NIM or the builder with `nsys` as container PID 1. NIM export dies in `TimeConversion.cpp`. There is no usable NIM `.nsys-rep`.
- GPU split is `config/pipeline.yaml` (`embed_index_ratio` `10:1`, `dynamic_embed: true`). **Memory** is capped. SMs are not. MPS lets both use every SM. Do not give both pods `nvidia.com/gpu: 1`. Do not treat MIG / SM partitioning as a win: embed already fills the card; a static 90/10 or 75/25 tax on embed for ~44 min is worse than embed-then-index. Open-source KAI cannot loan 10% SMs then give them back; `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` is process-lifetime and would slow the last flush.
- Hard caps need driver **r615+** and container toolkit **≥ 1.20.1**. `gpu_fraction_enforced` in `metrics.json` is that fact. `scripts/install-k8s-runai.sh` refuses any driver other than **615.71.10** and does not upgrade it. Cluster is already up; do not kubeadm init again.
- Builder image is `opensearchstaging/remote-vector-index-builder:api-snapshot` (cuVS 26.08, CUDA 12.9, `sm_120`). `api-latest` is CUDA 12.4 and fails on Blackwell with `cudaErrorInvalidValue` before any `.faiss` file.
- The builder process reads `AWS_DEFAULT_REGION` and `S3_ENDPOINT_URL`. Setting only `AWS_REGION` / `S3_ENDPOINT` fails every job in S3 client setup. OpenSearch then builds HNSW on CPU. For LocalStack, also set `AWS_REQUEST_CHECKSUM_CALCULATION=WHEN_REQUIRED` and `AWS_RESPONSE_CHECKSUM_VALIDATION=WHEN_REQUIRED`.
- A few-hundred-doc segment can fail CAGRA with "too many invalid or duplicated neighbor nodes" even on the snapshot image. `BULK_WORKERS=4` creates several Lucene segments per flush. Do not drop microbatch to 5k to "fix" SM sharing; the CAGRA/embed ratio stays the same and small graphs fail more often.
- `NIM_PERFORMANCE_MODE=1` OOMs at startup on this card when the builder also holds a memory reservation. Live NIM is `0`. Do not roll the NIM pod (hostPort 8000 deadlock). Builder strategy is Recreate (hostPort 1025).
- Keep `REMOTE_BUILD_POLL_INTERVAL=200ms`, `--flush-parallel 1`, and `MAX_WORKERS=1`.
- OpenSearch is CPU-only. From a container, the builder is the Docker bridge gateway on port 1025, not `127.0.0.1`. KAI admits per-container `gpu-memory.request` and `.limit` only. Do not also set `gpu-fraction` on the same pod. Live caps (local yaml, not the committed 96 GiB target): NIM 86016 MiB, builder 8898 MiB. NIM cpu 1–4, builder 500m–2 on this 8 vCPU node.
- `metrics.json` `gpu_sharing` text is the config-file story (embed limit 97887), not the live 86016/8898 caps. `s3_faiss_before/after` = -1 means listing failed, not "no build".
- NGC key is in `.env` (mode 600). Do not print or commit it.

## Commands

```bash
cp .env.example .env
python3 scripts/pipeline_config.py
./scripts/bootstrap.sh
# GPU CAGRA smoke / 1M overlap / 1M embed-then-index
./scripts/run_e2e.sh --limit 20000 --out-dir results/runai-smoke-20k
./scripts/run_e2e.sh --microbatch 10000 --flush-every 10000 --out-dir results/runai-1m-overlap
./scripts/run_e2e.sh --microbatch 1000000 --flush-every 0 --out-dir results/runai-1m-fullbatch
# CPU HNSW (builder not called)
./scripts/run_e2e.sh --limit 20000 --cpu-index --out-dir results/runai-smoke-20k-cpu-hnsw
./scripts/run_e2e.sh --microbatch 10000 --flush-every 10000 --cpu-index --out-dir results/runai-1m-overlap-cpu-hnsw
```

Unset `HTTP_PROXY`/`HTTPS_PROXY`. Single-node Run:ai install: `bash scripts/install-k8s-runai.sh`. Host cuVS for ground truth: `cuvs-cu13==26.8.1` and `cupy-cuda13x` from `pypi.nvidia.com`.

Nohup 1M (pick one; each deletes the index):

```bash
nohup ./scripts/run_e2e.sh --microbatch 10000 --flush-every 10000 --out-dir results/runai-1m-overlap-cuvs2608 > results/e2e-1m-overlap-cuvs2608.log 2>&1 &
nohup ./scripts/run_e2e.sh --microbatch 1000000 --flush-every 0 --out-dir results/runai-1m-fullbatch-cuvs2608 > results/e2e-1m-fullbatch-cuvs2608.log 2>&1 &
nohup ./scripts/run_e2e.sh --microbatch 10000 --flush-every 10000 --cpu-index --out-dir results/runai-1m-overlap-cpu-hnsw > results/e2e-1m-overlap-cpu-hnsw.log 2>&1 &
```

```bash
tail -f results/e2e-1m-overlap-cuvs2608.log | grep --line-buffered -E 'overlap |Indexed count|Error|Traceback'
```

## What already ran (do not rerun to "check")

NIM pod `nim-embed-5d58b85c6b-kbvbc` (hostPort 8000, perf mode 0). Builder `remote-index-builder-74b7cffd46-wx6nh` (`api-snapshot`, Recreate, `AWS_DEFAULT_REGION` + `S3_ENDPOINT_URL`). Driver **615.71.10**.

| Run | Wall | Embed | Bulk / flush | vec/s | Recall@10 | QPS / p95 |
|---|---:|---:|---:|---:|---:|---:|
| 10k overlap, CPU HNSW | 2762 s | 2630 s | 128 / 430 s | 362 | 0.9998 | 71.6 / 18.2 ms |
| Full embed, then GPU CAGRA | 2874 s | 2637 s | 221 / 17 s | 348 | 0.9996 | 36.1 / 36.2 ms |
| 10k overlap, GPU CAGRA | 3229 s | 2976 s | 143 / 2999 s | 310 | 0.99918 | 52.3 / 25.6 ms |

GPU CAGRA overlap: solo 10k embed 27 s (~91% util); overlapped median 29.7 s. Solo last flush 3.7 s; overlapped median ~30 s. `nvidia-smi` util is one clock (535/1583 samples at 100% on the GPU overlap). Logs: `results/e2e-1m-overlap-cuvs2608.log`, `results/e2e-1m-fullbatch-cuvs2608.log`, `results/e2e-1m-overlap-cpu-hnsw.log`.

## Nsys

Client trace only: `./scripts/nsys_trace_all.sh` → `results/nsys_all/e2e.nsys-rep`. NIM and the builder are not in that file. `--gpu-metrics-devices=all` often returns `ERR_NVGPUCTRPERM`. NVTX colors must be ints or built-in names (`blue`, `green`). Do not use `gray`.

`BULK_WORKERS=4` commits several Lucene segments per flush, so one `_flush` starts several serial CAGRA builds (`MAX_WORKERS=1`).

## Outputs

- `results/runai/metrics.json` unless `--out-dir` is set. A new run overwrites that file.
- Log lines `overlap embed|flush N started|finished` and `Index build: CPU FAISS HNSW` / `remote GPU CAGRA`.

OpenSearch JVM is CPU-only. Remote CAGRA is `remote-index-builder` on the same GPU as NIM. How-to: `notebooks/embed_index_runai.ipynb`.
