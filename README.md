# Embed and index on one GPU

Measure embed + CAGRA/HNSW index time, search p95, QPS, and Recall@10 for 1,000,000 MIRACL vectors on a single GPU. Embedding and indexing share that GPU through Run:ai's open-source [GPU Fractioning Operator](https://github.com/kai-scheduler/gpu-fractioning).

```
MIRACL texts → NIM (GPU fraction) → OpenSearch bulk
                 each microbatch → remote builder CAGRA (other fraction)
                 → cuVS GT top-10 → OpenSearch kNN → metrics.json
```

OpenSearch itself stays on CPU. The how-to notebook is [`notebooks/embed_index_runai.ipynb`](notebooks/embed_index_runai.ipynb).

## Configuration table

Edit [`config/pipeline.yaml`](config/pipeline.yaml), then restart the GPU containers. That file is the source of truth. Command-line flags override it for one run.

| Field | Default | Effect |
|---|---|---|
| `pipeline.embed_index_ratio` | `10:1` | Memory parts on one GPU. Embedding gets 10/11, indexing 1/11. |
| `pipeline.dynamic_embed` | `true` | Embedding's limit is the whole GPU, so idle index memory goes to embedding. The index cap stays at its 1 part. |
| `pipeline.microbatch_size` | `10000` | Vectors embedded, then indexed, per step. |
| `pipeline.index_every_docs` | `10000` | Remote CAGRA cadence. Same value as the microbatch means embed a batch, then index it, and repeat. `0` builds once after every bulk. |
| `dataset.vectors` | `1000000` | Corpus length. 1024-d FP32 is 4 GB. |
| `pipeline.target_recall_at_10` | `0.95` | Recorded in `metrics.json`. The run does not stop early on recall. |

Show the resolved MiB caps for the GPU this process can see:

```bash
python scripts/pipeline_config.py
```

The last line names the driver. `driver r615 enforces the caps` (or any major ≥ 615) means a hard memory cap is available. `gpu_fraction_enforced` is the same fact, and `run_e2e.py` copies it into `results/runai/metrics.json`.

On the target RTX PRO 6000 (96 GiB) the `10:1` row is embedding request 89368 MiB / limit 98304 MiB, and indexing request = limit 8936 MiB. A live Docker start uses GPU 0's real size instead of 96 GiB. The checked-in Kubernetes manifest stays on the 96 GiB target; regenerate it for a different card (below).

This repository configures the split. It does not install Run:ai, KAI Scheduler, or the GPU Fractioning Operator, and it does not upgrade the NVIDIA driver. The machine where this was written reports kernel module **610.43.02**, so a run there shares GPU 0 with the env vars set and no hard cap.

## Run on another machine

Use a host shell on the GPU machine. The agent sandbox cannot see `/dev/nvidia*` or the Docker socket.

Hard caps need all of the following. Docker alone sets the variables; the runtime and driver apply them.

- NVIDIA driver **r615 or newer** (CUDA 13.4). GPU Operator does not install r615 unless `driver.version` is set.
- NVIDIA Container Toolkit **v1.20.1 or newer** (`apply-cuda-memory-limits`).
- One GPU visible as GPU 0 (`NVIDIA_VISIBLE_DEVICES` in `.env`, default `0`).
- Docker Compose v2, an NGC API key, and either real S3 credentials or LocalStack (`AWS_ENDPOINT_URL` / `S3_ENDPOINT`).

```bash
git clone https://github.com/ManasSingh12345/embedding-indexing-poc
cd embedding-indexing-poc
cp .env.example .env          # NGC_API_KEY, S3_BUCKET, AWS keys or a LocalStack endpoint
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
python3 scripts/pipeline_config.py
./scripts/bootstrap.sh        # venv, MIRACL 1M, OpenSearch, builder, NIM
./scripts/run_e2e.sh --limit 20000
./scripts/run_e2e.sh          # 1,000,000 vectors, table defaults
```

`bootstrap.sh` creates `.venv`, installs `requirements.txt`, logs in to `nvcr.io`, downloads MIRACL, and calls `start_gpu_share.sh`. Run `./scripts/start_gpu_share.sh` again after any edit to `config/pipeline.yaml`. Each `run_e2e.sh` deletes and recreates the OpenSearch index `miracl-en-1m-cagra`.

`vm.max_map_count` must be at least 262144 (`bootstrap.sh` sets it when `sysctl` is allowed). NIM's first start can take about 15 minutes. cuVS and CuPy are installed by `run_e2e.sh` from `pypi.nvidia.com` (`cuvs-cu13==26.8.1`, `cupy-cuda13x`).

Embeddings are written as they finish to `results/runai/embeddings_fp32.npy` (1,000,000 × 1024 FP32). Metrics go to `results/runai/metrics.json`.

| Metric | Field |
|---|---|
| Embed + index time | `embed_index_wall_s` (`embed_s`, `bulk_s`, `flush_s` underneath) |
| Search p95 | `p95_latency_ms` |
| Average QPS | `avg_qps` |
| Recall@10 | `recall_at_10` |
| Caps actually enforced | `gpu_fraction_enforced` |

A proxy breaks localhost calls to NIM and OpenSearch. Leave `HTTP_PROXY` and `HTTPS_PROXY` unset in the shell that runs the scripts.

## Run:ai on Kubernetes

`deploy/k8s/embed-and-index.yaml` is the same 10:1 split as two pods on one GPU: NIM at `gpu-fraction: "0.9091"` (request 89368Mi, limit 98304Mi) and the builder at `"0.0909"` (8936Mi). KAI Scheduler places them. fractiond injects `NVIDIA_GPU_MEMORY_REQUEST` and `NVIDIA_GPU_MEMORY_LIMIT` before the container starts. Neither pod requests `nvidia.com/gpu`.

The manifest does not include OpenSearch. OpenSearch stays on CPU. Point `knn.remote_index_build.service.endpoint` at `http://remote-index-builder.embed-index.svc:1025`. `run_e2e.sh` still dials `127.0.0.1:8000`, `:9200`, and `:1025`, so those ports have to reach the pods from the client machine.

Cluster prerequisites, from the [GPU Fractioning Operator](https://github.com/kai-scheduler/gpu-fractioning) and [KAI GPU sharing](https://github.com/NVIDIA/KAI-Scheduler/blob/main/docs/gpu-sharing/README.md) docs:

- Kubernetes 1.28+
- containerd 2 with NRI, or CRI-O with NRI
- GPU Operator **v26.7.1** or newer, with the driver pinned to r615 (the chart default is older):

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update
helm install gpu-operator nvidia/gpu-operator \
  --version v26.7.1 \
  --namespace gpu-operator --create-namespace \
  --set driver.version=615.<patch>
```

Toolkit v1.20.1+ and device plugin v0.20.1+ ship with that operator version. Leave `cdi.nriPluginEnabled` false unless `gpu-fractioning` is added to `NRI_MANAGEMENT_CDI_DEVICE_NAMESPACES`.

- KAI Scheduler in NvFractions mode, which installs GPU fractioning with it:

```bash
helm upgrade -i kai-scheduler \
  oci://ghcr.io/kai-scheduler/kai-scheduler/kai-scheduler \
  -n kai-scheduler --create-namespace \
  --set global.nvFractions.set=true
```

Wait until `GpuFractioningConfig` `default` is Ready and the GPU node condition `gpu-fractioning.nvidia.com/Ready` is True. Chart flags move; if this install command is rejected, use the KAI GPU sharing doc linked above.

Then apply this repo's manifest. Create the secrets in `embed-index` before the pods start, and restart so they pick the secrets up:

```bash
kubectl apply -f deploy/k8s/embed-and-index.yaml
kubectl create secret generic ngc-api --from-literal=NGC_API_KEY="$NGC_API_KEY" -n embed-index
kubectl create secret generic s3-creds \
  --from-literal=AWS_REGION="$AWS_REGION" \
  --from-literal=AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  --from-literal=AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
  --from-literal=S3_BUCKET="$S3_BUCKET" \
  -n embed-index
kubectl rollout restart deployment -n embed-index
```

Regenerate after a table edit. The first file is the RTX PRO 6000 target and is what this repo commits. `--detect` sizes the caps to GPU 0 on the machine running the command; that file is gitignored.

```bash
python scripts/pipeline_config.py manifest --write deploy/k8s/embed-and-index.yaml
python scripts/pipeline_config.py manifest --detect --write deploy/k8s/embed-and-index.local.yaml
```

## Recall

Ground truth is cuVS brute-force cosine top-10 on the same L2-normalized vectors, not MIRACL qrels. Queries are document rows drawn with seed 42. OpenSearch search is inner product, `ef_search=256`, `k=10`.

Recall@10 matches cuVS `calc_recall`: for each query, `|set(pred[:k]) ∩ set(gt[:k])|`, divided by `n_queries * k`.

## Other knobs

These stay out of the ratio table because changing them does not change the GPU split. Override with flags or `.env`.

| Knob | Default | Why |
|---|---|---|
| `NIM_PERFORMANCE_MODE` | `1` | Throughput batch 64. Latency mode was about half the embed rate and used much less VRAM. |
| `REMOTE_BUILD_POLL_INTERVAL` | `200ms` | OpenSearch waits `3 × poll.interval` before the first status check. The 5s default is about 15s of idle time per flush. |
| `BULK_DOCS` / `BULK_WORKERS` | `1000` / `4` | Parallel `_bulk` posts. Four index threads can commit several Lucene segments on one flush, and each segment gets its own CAGRA build. |
| `FLUSH_PARALLEL` / `MAX_WORKERS` | `1` / `1` | One index, one CAGRA job. Several builds at once on one GPU were slower. |

If `AWS_ENDPOINT_URL` or `S3_ENDPOINT` is set, `start_stack.sh` starts LocalStack and creates the bucket.

## Not in git

`.env`, `data/`, `vendor/`, `results/`, `.venv/`, and `.cache/` stay local.

## Agent notes

Project skill: [`.cursor/skills/m1-e2e-baseline/SKILL.md`](.cursor/skills/m1-e2e-baseline/SKILL.md). OpenSearch remote-build settings: [`deploy/DEPLOYMENT.md`](deploy/DEPLOYMENT.md).
