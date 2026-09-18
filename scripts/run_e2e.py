#!/usr/bin/env python3
"""Sequential 1M embed → OpenSearch CAGRA_HNSW index, cuVS brute-force Recall@10."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests
from tqdm import tqdm

try:
    import nvtx
except ImportError:  # pragma: no cover
    from contextlib import nullcontext

    class nvtx:  # type: ignore[no-redef]
        @staticmethod
        def annotate(*_args: Any, **_kwargs: Any):
            return nullcontext()

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CUPY_CACHE_DIR", str(ROOT / ".cache" / "cupy"))
os.environ.setdefault("CUDA_CACHE_PATH", str(ROOT / ".cache" / "nv"))


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, eps)).astype(np.float32, copy=False)


class GpuSampler(threading.Thread):
    def __init__(self, out_csv: Path, interval_s: float = 2.0) -> None:
        super().__init__(daemon=True)
        self.out_csv = out_csv
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.util: list[float] = []
        self.mem_used_mib: list[float] = []

    def run(self) -> None:
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.out_csv.open("w", encoding="utf-8") as fh:
            fh.write("ts,util_gpu,mem_used_mib,mem_total_mib,processes\n")
            while not self.stop_event.is_set():
                try:
                    raw = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                    ).strip()
                    util, used, total = [float(x.strip()) for x in raw.split(",")]
                    procs = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-compute-apps=process_name,used_gpu_memory",
                            "--format=csv,noheader",
                        ],
                        text=True,
                    ).strip().replace("\n", " | ").replace(",", ";")
                    self.util.append(util)
                    self.mem_used_mib.append(used)
                    fh.write(f"{time.time():.3f},{util},{used},{total},{procs}\n")
                    fh.flush()
                except Exception as exc:  # noqa: BLE001
                    fh.write(f"{time.time():.3f},,,,{exc}\n")
                    fh.flush()
                self.stop_event.wait(self.interval_s)

    def stop(self) -> dict[str, float]:
        self.stop_event.set()
        self.join(timeout=5)
        return {
            "gpu_util_avg": float(statistics.mean(self.util)) if self.util else 0.0,
            "gpu_util_max": float(max(self.util)) if self.util else 0.0,
            "gpu_mem_used_mib_max": float(max(self.mem_used_mib)) if self.mem_used_mib else 0.0,
        }


class NimClient:
    def __init__(self, url: str, model: str, dim: int, batch_size: int) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self.session = requests.Session()
        self.tokens = 0

    def embed(self, texts: list[str], input_type: str) -> np.ndarray:
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            payload = {
                "input": chunk,
                "model": self.model,
                "input_type": input_type,
                "modality": "text",
                "embedding_type": "float",
                "encoding_format": "float",
                "dimensions": self.dim,
            }
            last_exc: Exception | None = None
            for attempt in range(6):
                try:
                    resp = self.session.post(
                        f"{self.url}/v1/embeddings",
                        json=payload,
                        timeout=180,
                    )
                    if resp.status_code >= 400 and "dimension" in resp.text.lower() and "dimensions" in payload:
                        payload.pop("dimensions", None)
                        continue
                    if resp.status_code >= 400:
                        raise RuntimeError(f"NIM {resp.status_code}: {resp.text[:500]}")
                    body = resp.json()
                    rows = sorted(body["data"], key=lambda r: r["index"])
                    vecs = np.asarray([r["embedding"] for r in rows], dtype=np.float32)
                    if vecs.shape[1] != self.dim:
                        vecs = vecs[:, : self.dim]
                    out[start : start + len(chunk)] = vecs
                    usage = body.get("usage") or {}
                    self.tokens += int(usage.get("total_tokens") or usage.get("prompt_tokens") or 0)
                    last_exc = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    time.sleep(min(2**attempt, 16))
            if last_exc is not None:
                raise last_exc
        return l2_normalize(out)


class OpenSearchKnn:
    def __init__(self, url: str, index: str, dim: int) -> None:
        self.url = url.rstrip("/")
        self.index = index
        self.dim = dim
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _req(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        resp = self.session.request(method, f"{self.url}{path}", timeout=kwargs.pop("timeout", 120), **kwargs)
        if resp.status_code >= 400:
            raise RuntimeError(f"OpenSearch {method} {path} {resp.status_code}: {resp.text[:800]}")
        return resp

    def recreate(self) -> None:
        exists = self.session.head(f"{self.url}/{self.index}", timeout=30)
        if exists.status_code == 200:
            self._req("DELETE", f"/{self.index}")
        body = {
            "settings": {
                "index.knn": True,
                "index.knn.remote_index_build.enabled": True,
                "index.knn.remote_index_build.size.min": "1kb",
                "index.knn.advanced.approximate_threshold": 0,
                "index.knn.algo_param.ef_search": 256,
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "refresh_interval": "-1",
            },
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "knn_vector",
                        "dimension": self.dim,
                        "method": {
                            "name": "hnsw",
                            "engine": "faiss",
                            "space_type": "innerproduct",
                            "parameters": {"m": 16, "ef_construction": 256},
                        },
                    },
                    "docid": {"type": "keyword"},
                    "row": {"type": "integer"},
                }
            },
        }
        self._req("PUT", f"/{self.index}", json=body)

    def bulk_vectors(self, rows: np.ndarray, vectors: np.ndarray, docids: list[str], bulk_docs: int) -> None:
        for start in range(0, len(rows), bulk_docs):
            sl = slice(start, start + bulk_docs)
            lines: list[str] = []
            for row, vec, docid in zip(rows[sl], vectors[sl], docids[sl], strict=True):
                lines.append(json.dumps({"index": {"_index": self.index, "_id": str(int(row))}}))
                lines.append(
                    json.dumps(
                        {
                            "vector": vec.tolist(),
                            "docid": docid,
                            "row": int(row),
                        }
                    )
                )
            payload = ("\n".join(lines) + "\n").encode("utf-8")
            resp = self.session.post(
                f"{self.url}/_bulk",
                data=payload,
                headers={"Content-Type": "application/x-ndjson"},
                timeout=300,
            )
            if resp.status_code >= 400:
                raise RuntimeError(f"bulk {resp.status_code}: {resp.text[:500]}")
            body = resp.json()
            if body.get("errors"):
                err = next(
                    item["index"]["error"]
                    for item in body["items"]
                    if "error" in item.get("index", {})
                )
                raise RuntimeError(f"bulk item error: {err}")

    def flush_and_refresh(self) -> None:
        self._req("POST", f"/{self.index}/_flush", timeout=300)
        self._req("POST", f"/{self.index}/_refresh", timeout=120)

    def count(self) -> int:
        return int(self._req("GET", f"/{self.index}/_count").json()["count"])

    def knn_stats(self) -> dict[str, Any]:
        try:
            return self._req("GET", "/_plugins/_knn/stats").json()
        except Exception:  # noqa: BLE001
            return {}

    def search(self, vector: np.ndarray, k: int = 10, ef_search: int = 256) -> list[int]:
        knn_field: dict[str, Any] = {
            "vector": vector.tolist(),
            "k": k,
            "method_parameters": {"ef_search": ef_search},
        }
        body: dict[str, Any] = {"size": k, "_source": ["row"], "query": {"knn": {"vector": knn_field}}}
        try:
            hits = self._req("POST", f"/{self.index}/_search", json=body, timeout=60).json()["hits"]["hits"]
        except RuntimeError:
            knn_field.pop("method_parameters", None)
            hits = self._req("POST", f"/{self.index}/_search", json=body, timeout=60).json()["hits"]["hits"]
        rows = []
        for hit in hits:
            src = hit.get("_source") or {}
            if "row" in src:
                rows.append(int(src["row"]))
            else:
                rows.append(int(hit["_id"]))
        return rows


def wait_for_remote_build(os_knn: OpenSearchKnn, before: dict[str, Any], timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        after = os_knn.knn_stats()
        # If stats expose counters, wait until they move. Otherwise a short settle is enough
        # once flush returned — CAGRA on 10k x 1024 is typically a few seconds.
        if after and after != before:
            return
        time.sleep(2)
    # Do not fail the run solely on opaque stats; flush already completed.


def load_docs(path: Path, limit: int) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    docids: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            title = (rec.get("title") or "").strip()
            body = (rec.get("text") or "").strip()
            text = f"{title}\n{body}".strip() if title and body else (title or body)
            if not text:
                continue
            texts.append(text)
            docids.append(str(rec.get("docid") or len(docids)))
            if len(texts) >= limit:
                break
    return texts, docids


def recall_at_k(gt: np.ndarray, pred: list[list[int]], k: int) -> float:
    hits = 0
    for i, neigh in enumerate(pred):
        hits += len(set(gt[i, :k].tolist()) & set(neigh[:k]))
    return hits / float(len(pred) * k)


def cuvs_bruteforce_topk(base: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    import cupy as cp
    from cuvs.neighbors import brute_force

    base_gpu = cp.asarray(base, dtype=cp.float32)
    queries_gpu = cp.asarray(queries, dtype=cp.float32)
    index = brute_force.build(base_gpu, metric="cosine")
    _distances, neighbors = brute_force.search(index, queries_gpu, k=k)
    return cp.asnumpy(neighbors).astype(np.int64)


def s3_faiss_count(bucket: str, prefix: str, region: str) -> int:
    if not bucket:
        return -1
    try:
        import boto3

        s3 = boto3.client("s3", region_name=region)
        token = None
        n = 0
        while True:
            kwargs = {"Bucket": bucket, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            resp = s3.list_objects_v2(**kwargs)
            n += sum(1 for obj in resp.get("Contents", []) if obj["Key"].endswith(".faiss"))
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
        return n
    except Exception:  # noqa: BLE001
        return -1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1_000_000)
    parser.add_argument("--microbatch", type=int, default=10_000)
    parser.add_argument("--nim-batch", type=int, default=64)
    parser.add_argument("--bulk-docs", type=int, default=250)
    parser.add_argument("--queries", type=int, default=5_000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--ef-search", type=int, default=256)
    parser.add_argument("--gpu-usd-per-hour", type=float, default=3.0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "m1_sequential",
    )
    args = parser.parse_args()

    docs_path = ROOT / "data" / "miracl-en-1m" / "docs.jsonl"
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / "embeddings_fp32.npy"

    nim_url = os.environ.get("NIM_URL", f"http://127.0.0.1:{os.environ.get('NIM_PORT', '8000')}")
    os_url = os.environ.get("OPENSEARCH_URL", "http://127.0.0.1:9200")
    model = os.environ.get("NIM_MODEL_NAME", "nvidia/llama-nemotron-embed-vl-1b-v2")
    index_name = os.environ.get("OPENSEARCH_INDEX", "miracl-en-1m-cagra")
    bucket = os.environ.get("S3_BUCKET", "")
    region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION") or "us-east-2"
    prefix = os.environ.get("S3_PREFIX", "knn-indexes")

    print(f"Loading {args.limit} MIRACL docs from {docs_path}", flush=True)
    with nvtx.annotate("load_docs", color=0x808080):
        texts, docids = load_docs(docs_path, args.limit)
    n = len(texts)
    print(f"Loaded {n} documents", flush=True)

    embeddings = np.lib.format.open_memmap(
        emb_path, mode="w+", dtype=np.float32, shape=(n, args.dim)
    )

    nim = NimClient(nim_url, model, args.dim, args.nim_batch)
    os_knn = OpenSearchKnn(os_url, index_name, args.dim)
    print("Creating OpenSearch index", flush=True)
    with nvtx.annotate("opensearch_recreate", color=0xFFD700):
        os_knn.recreate()

    gpu = GpuSampler(out_dir / "gpu_timeseries.csv")
    gpu.start()

    embed_s = 0.0
    index_s = 0.0
    t0 = time.perf_counter()
    n_batches = (n + args.microbatch - 1) // args.microbatch
    faiss_before = s3_faiss_count(bucket, prefix, region)

    try:
        for b in tqdm(range(n_batches), desc="microbatches"):
            sl = slice(b * args.microbatch, min(n, (b + 1) * args.microbatch))
            batch_texts = texts[sl]
            batch_ids = docids[sl]
            rows = np.arange(sl.start, sl.stop, dtype=np.int32)

            te0 = time.perf_counter()
            with nvtx.annotate("embed_batch", color=0x1E90FF):
                vecs = nim.embed(batch_texts, input_type="passage")
                embeddings[sl] = vecs
                embeddings.flush()
            embed_s += time.perf_counter() - te0

            ti0 = time.perf_counter()
            with nvtx.annotate("index_batch", color=0x32CD32):
                stats_before = os_knn.knn_stats()
                os_knn.bulk_vectors(rows, vecs, batch_ids, args.bulk_docs)
                os_knn.flush_and_refresh()
                time.sleep(3)
                wait_for_remote_build(os_knn, stats_before, timeout_s=180)
            index_s += time.perf_counter() - ti0
    finally:
        gpu_summary = gpu.stop()

    wall_s = time.perf_counter() - t0
    count = os_knn.count()
    faiss_after = s3_faiss_count(bucket, prefix, region)

    print(f"Indexed count={count} wall={wall_s:.1f}s embed={embed_s:.1f}s index={index_s:.1f}s", flush=True)

    rng = random.Random(42)
    qn = min(args.queries, n)
    query_rows = np.array(rng.sample(range(n), qn), dtype=np.int64)
    query_vecs = np.asarray(embeddings[query_rows], dtype=np.float32)

    print(f"cuVS brute-force GT top-{args.k} for {qn} queries", flush=True)
    gt_t0 = time.perf_counter()
    with nvtx.annotate("cuvs_bruteforce_topk10", color=0xDC143C):
        base = np.asarray(embeddings, dtype=np.float32)
        gt = cuvs_bruteforce_topk(base, query_vecs, args.k)
    gt_s = time.perf_counter() - gt_t0
    np.save(out_dir / "gt_neighbors.npy", gt)
    np.save(out_dir / "query_rows.npy", query_rows)

    print("OpenSearch kNN search for recall + latency", flush=True)
    with nvtx.annotate("opensearch_search", color=0x9370DB):
        for vec in query_vecs[: min(20, qn)]:
            os_knn.search(vec, k=args.k, ef_search=args.ef_search)

        pred = []
        lat_s = []
        search_t0 = time.perf_counter()
        for vec in tqdm(query_vecs, desc="search"):
            s0 = time.perf_counter()
            pred.append(os_knn.search(vec, k=args.k, ef_search=args.ef_search))
            lat_s.append(time.perf_counter() - s0)
        search_wall = time.perf_counter() - search_t0
    recall = recall_at_k(gt, pred, args.k)
    lat_ms = [x * 1000.0 for x in lat_s]
    p95 = float(np.percentile(lat_ms, 95))
    p50 = float(np.percentile(lat_ms, 50))
    qps = qn / search_wall if search_wall > 0 else 0.0

    usd = (wall_s / 3600.0) * args.gpu_usd_per_hour
    results = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "gpu_sharing": "sequential embed-then-index, both containers on GPU 0, no Run:ai",
        "model": model,
        "index": index_name,
        "dim": args.dim,
        "dtype": "FP32",
        "metric": "innerproduct (L2-normalized = cosine)",
        "n_docs": n,
        "microbatch": args.microbatch,
        "nim_batch": args.nim_batch,
        "k": args.k,
        "ef_search": args.ef_search,
        "n_queries": qn,
        "embed_index_wall_s": wall_s,
        "embed_s": embed_s,
        "index_s": index_s,
        "embed_index_ratio": (embed_s / index_s) if index_s else None,
        "vectors_per_s": n / wall_s if wall_s else None,
        "embed_vectors_per_s": n / embed_s if embed_s else None,
        "embed_tokens": nim.tokens,
        "opensearch_count": count,
        "s3_faiss_before": faiss_before,
        "s3_faiss_after": faiss_after,
        "cuvs_gt_s": gt_s,
        "recall_at_10": recall,
        "target_recall_at_10": 0.95,
        "avg_qps": qps,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "gpu_usd_per_hour": args.gpu_usd_per_hour,
        "gpu_usd_per_hour_note": "Override with --gpu-usd-per-hour; default is a placeholder.",
        "cost_usd_for_1m": usd,
        "vectors_per_dollar": n / usd if usd else None,
        **gpu_summary,
    }
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    if count != n:
        print(f"WARNING: OpenSearch count {count} != {n}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
