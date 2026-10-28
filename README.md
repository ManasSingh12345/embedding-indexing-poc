# Embed and index on one GPU

Measure embed + index wall time, search p95, QPS, and Recall@10 for 1,000,000 MIRACL vectors (1024-d FP32) on one GPU. NIM and the remote index builder share that GPU through Run:ai's [GPU Fractioning Operator](https://github.com/kai-scheduler/gpu-fractioning). OpenSearch stays on CPU.

`scripts/run_e2e.py` embeds microbatch N+1 while flush N runs. Only one remote build is in flight. The served index is FAISS HNSW (`HNSW16,Flat`, inner product). CAGRA is the GPU build, not a separate search index. `--cpu-index` skips the builder and builds that HNSW on the OpenSearch JVM.

```
MIRACL texts → NIM → OpenSearch bulk
                next embed overlaps flush N (CAGRA or CPU HNSW)
                → cuVS GT top-10 → OpenSearch kNN → metrics.json
```

How-to: [`notebooks/embed_index_runai.ipynb`](notebooks/embed_index_runai.ipynb). Source of truth: [`config/pipeline.yaml`](config/pipeline.yaml).

## Configuration

| Field | Default | Effect |
|---|---|---|
| `pipeline.embed_index_ratio` | `10:1` | **Memory** split. Embedding 10/11, indexing 1/11. SMs are not split. |
| `pipeline.dynamic_embed` | `true` | Embedding's limit is the whole GPU. The index cap stays at its 1 part. |
| `pipeline.microbatch_size` | `10000` | Vectors per embed step. |
| `pipeline.index_every_docs` | `10000` | Flush cadence. `0` builds once after every bulk. |
| `dataset.vectors` | `1000000` | Corpus length. |
| `pipeline.target_recall_at_10` | `0.95` | Recorded in metrics. The run does not stop on recall. |

```bash
python scripts/pipeline_config.py
```

`driver r615 enforces the caps` means the memory cap is real. On a 96 GiB RTX PRO 6000 the 10:1 row is embed request 89368 MiB / limit 98304 MiB, index request = limit 8936 MiB. `--detect` uses GPU 0's real size. The committed manifest stays on the 96 GiB target. Live pods on the 8 vCPU box use lower CPU/memory and a slightly lower embed MiB cap so the builder can schedule; that lives in gitignored `embed-and-index.local.yaml`.

Run:ai fractions do not partition SMs. Both processes may use every SM (MPS). Overlapping GPU CAGRA with a saturated embedder slows embed ~10% and stretches CAGRA ~8×. A hard SM split (MIG, 25% steps on this SKU) is worse for this pipeline: embed wants the whole GPU for ~44 min, then a short index. `--flush-every 0` is embed then one GPU index. `--cpu-index` is GPU embed plus CPU HNSW.

## Run

Host shell on the GPU machine. Unset `HTTP_PROXY` and `HTTPS_PROXY`. Each `run_e2e.sh` deletes and recreates `miracl-en-1m-cagra`. Do not start a second e2e while one is running.

```bash
cp .env.example .env
python3 scripts/pipeline_config.py
./scripts/bootstrap.sh
./scripts/run_e2e.sh --limit 20000 --out-dir results/runai-smoke-20k
./scripts/run_e2e.sh --limit 20000 --cpu-index --out-dir results/runai-smoke-20k-cpu-hnsw
./scripts/run_e2e.sh --microbatch 10000 --flush-every 10000 --out-dir results/runai-1m-overlap
./scripts/run_e2e.sh --microbatch 1000000 --flush-every 0 --out-dir results/runai-1m-fullbatch
./scripts/run_e2e.sh --microbatch 10000 --flush-every 10000 --cpu-index --out-dir results/runai-1m-overlap-cpu-hnsw
```

`--flush-every 0 --microbatch 1000000` is one embed, then one index. `--cpu-index` sets `index.knn.remote_index_build.enabled` false for that index. Metrics land in `results/runai/metrics.json` unless `--out-dir` is set. `embed_index_wall_s` is the overlapped schedule; `embed_s` is the sum of embed calls. `index_build` is `gpu_cagra` or `cpu_hnsw`.

| Metric | Field |
|---|---|
| Embed + index time | `embed_index_wall_s` (`embed_s`, `bulk_s`, `flush_s`) |
| Search p95 / QPS | `p95_latency_ms`, `avg_qps` |
| Recall@10 | `recall_at_10` |
| Index path | `index_build` |
| Caps enforced | `gpu_fraction_enforced` |

Ground truth is cuVS brute-force cosine top-10 on the indexed vectors, not MIRACL qrels. Recall@10 is `|pred ∩ gt| / (n_queries * k)`.

On this host (NIM warm, snapshot builder, 1M docs) the shortest walls were CPU-HNSW overlap (~46 min, recall 0.9998) then full embed + one GPU CAGRA (~48 min, 0.9996). 10k GPU-CAGRA overlap was ~54 min (0.99918): embed dropped from ~379 to ~336 vec/s while CAGRA shared SMs.

## Builder image

`opensearchproject/remote-vector-index-builder:api-latest` is CUDA 12.4 / cuVS 25.08. On Blackwell (compute capability 12.0) every CAGRA build fails with `cudaErrorInvalidValue`. Use `opensearchstaging/remote-vector-index-builder:api-snapshot` (cuVS 26.08, CUDA 12.9, `sm_120`). That is the default in Compose and in `deploy/k8s/embed-and-index.yaml`.

The builder reads `AWS_DEFAULT_REGION` and `S3_ENDPOINT_URL`. `AWS_REGION` and `S3_ENDPOINT` are not enough. Compose sets both. A segment of a few hundred vectors can still fail with "too many invalid or duplicated neighbor nodes"; larger segments from the same flush succeed. Four bulk workers leave several Lucene segments per flush, and each gets its own build.

`NIM_PERFORMANCE_MODE=1` is the throughput default. On this shared, memory-capped GPU it allocates a large cuDNN workspace at startup and the embedder OOMs. Set it to `0` when NIM and the builder are both reserved on the card.

## Kubernetes

`scripts/install-k8s-runai.sh` brings up a single-node cluster: containerd NRI, kubeadm, Flannel, GPU Operator v26.7.1 with the host driver left as-is, and KAI Scheduler NvFractions. It refuses to run if `nvidia-smi` is not **615.71.10**. It writes `deploy/k8s/embed-and-index.local.yaml` (gitignored) with hostPorts 8000 and 1025 and the LocalStack endpoint. OpenSearch and LocalStack stay in Docker. Do not re-run kubeadm if the cluster is already up. Do not replace the driver.

```bash
bash scripts/install-k8s-runai.sh
```

Apply the committed manifest on a machine that already has the cluster:

```bash
kubectl apply -f deploy/k8s/embed-and-index.yaml
```

Create `ngc-api` and `s3-creds` in namespace `embed-index` first. Point OpenSearch `knn.remote_index_build.service.endpoint` at a URL the OpenSearch container can reach (the Docker bridge gateway, port 1025), not `127.0.0.1`. `run_e2e.sh` dials `127.0.0.1:8000`, `:9200`, and `:1025`.

Regenerate after a table edit:

```bash
python scripts/pipeline_config.py manifest --write deploy/k8s/embed-and-index.yaml
python scripts/pipeline_config.py manifest --detect --write deploy/k8s/embed-and-index.local.yaml
```

## Other knobs

| Knob | Default | Why |
|---|---|---|
| `REMOTE_BUILD_POLL_INTERVAL` | `200ms` | The 5s default idles about 15s per flush. |
| `BULK_DOCS` / `BULK_WORKERS` | `1000` / `4` | Parallel bulk. Several Lucene segments per flush. |
| `FLUSH_PARALLEL` / `MAX_WORKERS` | `1` / `1` | One remote build. Parallel builds on one GPU were slower. |
| `--cpu-index` | off | CPU FAISS HNSW. Builder stays up but is not called. |

## Not in git

`.env`, `data/`, `vendor/`, `results/`, `.venv/`, `.cache/`, `deploy/k8s/*.local.yaml`, and `images/`.

Project skill: [`.cursor/skills/m1-e2e-baseline/SKILL.md`](.cursor/skills/m1-e2e-baseline/SKILL.md). OpenSearch remote-build settings: [`deploy/DEPLOYMENT.md`](deploy/DEPLOYMENT.md).
