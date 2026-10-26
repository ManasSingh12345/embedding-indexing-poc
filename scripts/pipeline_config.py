#!/usr/bin/env python3
"""Load config/pipeline.yaml and turn the embed:index ratio into GPU fractions.

Run:ai's open-source GPU Fractioning Operator (kai-scheduler/gpu-fractioning)
injects NVIDIA_GPU_MEMORY_REQUEST and NVIDIA_GPU_MEMORY_LIMIT, in MiB, before
a container starts. Docker on this host sets those same variables. Kubernetes
manifests carry the matching annotations. The driver enforces the limit when
it is r615 / CUDA 13.4 or newer.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "pipeline.yaml"
MIN_DRIVER_MAJOR = 615


@dataclass(frozen=True)
class GpuShare:
    ratio: str
    dynamic_embed: bool
    total_mib: int
    gpu_name: str
    driver_major: int | None
    embed_request_mib: int
    embed_limit_mib: int
    index_request_mib: int
    index_limit_mib: int
    embed_fraction: float
    index_fraction: float

    @property
    def enforced(self) -> bool:
        return self.driver_major is not None and self.driver_major >= MIN_DRIVER_MAJOR


def load(path: Path | None = None) -> dict:
    with (path or CONFIG_PATH).open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path or CONFIG_PATH} must be a mapping")
    return data


def parse_ratio(ratio: str) -> tuple[int, int]:
    embed_s, index_s = str(ratio).split(":", 1)
    embed, index = int(embed_s), int(index_s)
    if embed < 1 or index < 1:
        raise ValueError(f"embed_index_ratio must be positive parts, got {ratio}")
    return embed, index


def _nvidia_smi(query: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = next((ln.strip() for ln in out.splitlines() if ln.strip()), "")
    return line or None


def detect_gpu(cfg: dict) -> tuple[int, str, int | None]:
    """Memory and name for GPU 0. Falls back to the target GPU in the table."""
    hw = cfg["hardware"]
    fallback = int(hw["target_gpu_memory_mib"])
    name = str(hw["target_gpu"])
    raw = _nvidia_smi("memory.total,name")
    if raw and "," in raw:
        mem_s, name = [part.strip() for part in raw.split(",", 1)]
        try:
            fallback = int(float(mem_s))
        except ValueError:
            pass
    driver = _nvidia_smi("driver_version")
    major = None
    if driver:
        head = driver.split(".", 1)[0]
        if head.isdigit():
            major = int(head)
    return fallback, name, major


def resolve(
    cfg: dict,
    *,
    memory_mib: int | None = None,
    gpu_name: str | None = None,
    use_live_driver: bool = True,
) -> GpuShare:
    pipeline = cfg["pipeline"]
    if memory_mib is None:
        memory_mib, detected_name, driver_major = detect_gpu(cfg)
        gpu_name = gpu_name or detected_name
    else:
        driver_major = detect_gpu(cfg)[2] if use_live_driver else None
        gpu_name = gpu_name or str(cfg["hardware"]["target_gpu"])
    embed_parts, index_parts = parse_ratio(pipeline["embed_index_ratio"])
    parts = embed_parts + index_parts
    index_request = max(1, memory_mib * index_parts // parts)
    embed_request = memory_mib - index_request
    dynamic = bool(pipeline["dynamic_embed"])
    embed_limit = memory_mib if dynamic else embed_request
    return GpuShare(
        ratio=str(pipeline["embed_index_ratio"]),
        dynamic_embed=dynamic,
        total_mib=memory_mib,
        gpu_name=gpu_name,
        driver_major=driver_major,
        embed_request_mib=embed_request,
        embed_limit_mib=embed_limit,
        index_request_mib=index_request,
        index_limit_mib=index_request,
        embed_fraction=embed_request / memory_mib,
        index_fraction=index_request / memory_mib,
    )


def apply_run_args(args: Any):
    """Fill unset CLI values from the table. Environment overrides the table."""
    cfg = load()
    pipeline = cfg["pipeline"]
    dataset = cfg["dataset"]
    index = cfg["index"]

    def from_env(cli: int | None, env_key: str, table_value: int) -> int:
        if cli is not None:
            return cli
        raw = os_environ_get(env_key)
        return int(raw) if raw else int(table_value)

    if args.limit is None:
        args.limit = int(dataset["vectors"])
    if args.microbatch is None:
        args.microbatch = int(pipeline["microbatch_size"])
    if args.flush_every is None:
        args.flush_every = int(pipeline["index_every_docs"])
    if args.queries is None:
        args.queries = int(pipeline["queries"])
    if args.nim_batch is None:
        args.nim_batch = int(pipeline["nim_batch"])
    args.bulk_docs = from_env(args.bulk_docs, "BULK_DOCS", int(pipeline["bulk_docs"]))
    args.bulk_workers = from_env(args.bulk_workers, "BULK_WORKERS", int(pipeline["bulk_workers"]))
    if args.k is None:
        args.k = int(pipeline["k"])
    if args.dim is None:
        args.dim = int(dataset["dim"])
    if args.ef_search is None:
        args.ef_search = int(index["ef_search"])
    if args.out_dir is None:
        args.out_dir = ROOT / str(pipeline["output_dir"])
    return cfg


def os_environ_get(key: str) -> str:
    import os

    return os.environ.get(key, "").strip()


def share_comment(share: GpuShare) -> str:
    dynamic = (
        "embedding limit is the whole GPU while indexing is idle"
        if share.dynamic_embed
        else "both limits equal their requests"
    )
    driver = (
        f"driver r{share.driver_major} enforces the caps"
        if share.enforced
        else (
            f"driver r{share.driver_major} does not enforce caps (need r{MIN_DRIVER_MAJOR}+)"
            if share.driver_major is not None
            else f"caps are enforced by driver r{MIN_DRIVER_MAJOR}+ (CUDA 13.4)"
        )
    )
    return (
        f"Run:ai {share.ratio} on {share.gpu_name} ({share.total_mib} MiB): "
        f"embed request {share.embed_request_mib} MiB limit {share.embed_limit_mib} MiB, "
        f"index request=limit {share.index_request_mib} MiB; {dynamic}; {driver}"
    )


def export_lines(share: GpuShare) -> str:
    values = {
        "EMBED_GPU_MEMORY_REQUEST_MIB": share.embed_request_mib,
        "EMBED_GPU_MEMORY_LIMIT_MIB": share.embed_limit_mib,
        "INDEX_GPU_MEMORY_REQUEST_MIB": share.index_request_mib,
        "INDEX_GPU_MEMORY_LIMIT_MIB": share.index_limit_mib,
        "EMBED_GPU_FRACTION": f"{share.embed_fraction:.4f}",
        "INDEX_GPU_FRACTION": f"{share.index_fraction:.4f}",
    }
    return "\n".join(f"export {key}={value}" for key, value in values.items()) + "\n"


def render_manifest(cfg: dict, share: GpuShare) -> str:
    runai = cfg["runai"]
    pipeline = cfg["pipeline"]
    ns = runai["namespace"]
    queue = runai["queue"]
    scheduler = runai["scheduler"]
    embed_req = f"{share.embed_request_mib}Mi"
    embed_lim = f"{share.embed_limit_mib}Mi"
    index_mib = f"{share.index_request_mib}Mi"
    return f"""# Generated from config/pipeline.yaml by scripts/pipeline_config.py.
# GPU Fractioning Operator: https://github.com/kai-scheduler/gpu-fractioning
# {share_comment(share)}
# OpenSearch stays on CPU. Point knn.remote_index_build.service.endpoint
# at http://remote-index-builder.{ns}.svc:1025
apiVersion: v1
kind: Namespace
metadata:
  name: {ns}
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nim-embed
  namespace: {ns}
  labels:
    app: nim-embed
    kai.scheduler/queue: {queue}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: nim-embed
  template:
    metadata:
      labels:
        app: nim-embed
        kai.scheduler/queue: {queue}
      annotations:
        gpu-fraction: "{share.embed_fraction:.4f}"
        nvidia.com/container.nim.gpu-memory.request: {embed_req}
        nvidia.com/container.nim.gpu-memory.limit: {embed_lim}
    spec:
      schedulerName: {scheduler}
      containers:
        - name: nim
          image: {pipeline["nim_image"]}
          ports:
            - containerPort: 8000
          env:
            - name: NGC_API_KEY
              valueFrom:
                secretKeyRef:
                  name: ngc-api
                  key: NGC_API_KEY
            - name: NIM_PERFORMANCE_MODE
              value: "{pipeline["nim_performance_mode"]}"
            - name: NVIDIA_GPU_MEMORY_REQUEST
              value: "{share.embed_request_mib}"
            - name: NVIDIA_GPU_MEMORY_LIMIT
              value: "{share.embed_limit_mib}"
          resources:
            requests:
              cpu: "8"
              memory: 32Gi
            limits:
              memory: 64Gi
---
apiVersion: v1
kind: Service
metadata:
  name: nim-embed
  namespace: {ns}
spec:
  selector:
    app: nim-embed
  ports:
    - port: 8000
      targetPort: 8000
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: remote-index-builder
  namespace: {ns}
  labels:
    app: remote-index-builder
    kai.scheduler/queue: {queue}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: remote-index-builder
  template:
    metadata:
      labels:
        app: remote-index-builder
        kai.scheduler/queue: {queue}
      annotations:
        gpu-fraction: "{share.index_fraction:.4f}"
        nvidia.com/container.builder.gpu-memory.request: {index_mib}
        nvidia.com/container.builder.gpu-memory.limit: {index_mib}
    spec:
      schedulerName: {scheduler}
      containers:
        - name: builder
          image: opensearchstaging/remote-vector-index-builder:api-snapshot
          imagePullPolicy: Always
          ports:
            - containerPort: 1025
          env:
            - name: MAX_WORKERS
              value: "1"
            - name: NVIDIA_GPU_MEMORY_REQUEST
              value: "{share.index_request_mib}"
            - name: NVIDIA_GPU_MEMORY_LIMIT
              value: "{share.index_limit_mib}"
            - name: AWS_REGION
              valueFrom:
                secretKeyRef:
                  name: s3-creds
                  key: AWS_REGION
            - name: AWS_DEFAULT_REGION
              valueFrom:
                secretKeyRef:
                  name: s3-creds
                  key: AWS_REGION
            - name: AWS_ACCESS_KEY_ID
              valueFrom:
                secretKeyRef:
                  name: s3-creds
                  key: AWS_ACCESS_KEY_ID
            - name: AWS_SECRET_ACCESS_KEY
              valueFrom:
                secretKeyRef:
                  name: s3-creds
                  key: AWS_SECRET_ACCESS_KEY
            - name: S3_BUCKET
              valueFrom:
                secretKeyRef:
                  name: s3-creds
                  key: S3_BUCKET
          resources:
            requests:
              cpu: "4"
              memory: 8Gi
            limits:
              memory: 16Gi
---
apiVersion: v1
kind: Service
metadata:
  name: remote-index-builder
  namespace: {ns}
spec:
  selector:
    app: remote-index-builder
  ports:
    - port: 1025
      targetPort: 1025
"""


def print_table(cfg: dict, share: GpuShare) -> None:
    pipeline = cfg["pipeline"]
    rows = [
        ("dataset", cfg["dataset"]["name"]),
        ("vectors", cfg["dataset"]["vectors"]),
        ("dim", cfg["dataset"]["dim"]),
        ("dtype", cfg["dataset"]["dtype"]),
        ("dataset_gb", cfg["dataset"]["size_gb"]),
        ("index", cfg["index"]["type"]),
        ("index_gb", cfg["index"]["size_gb"]),
        ("model", pipeline["embedding_model"]),
        ("vector_db", pipeline["vector_db"]),
        ("embed_index_ratio", pipeline["embed_index_ratio"]),
        ("dynamic_embed", pipeline["dynamic_embed"]),
        ("microbatch_size", pipeline["microbatch_size"]),
        ("index_every_docs", pipeline["index_every_docs"]),
        ("target_recall_at_10", pipeline["target_recall_at_10"]),
        ("target_gpu", cfg["hardware"]["target_gpu"]),
        ("resolved_gpu", share.gpu_name),
        ("driver_major", share.driver_major if share.driver_major is not None else "unread"),
        ("gpu_fraction_enforced", share.enforced),
        ("embed_request_mib", share.embed_request_mib),
        ("embed_limit_mib", share.embed_limit_mib),
        ("index_request_mib", share.index_request_mib),
        ("index_limit_mib", share.index_limit_mib),
    ]
    width = max(len(str(name)) for name, _ in rows)
    print(f"{'field'.ljust(width)}  value")
    print(f"{'-' * width}  -----")
    for name, value in rows:
        print(f"{str(name).ljust(width)}  {value}")
    print()
    print(share_comment(share))


def main(argv: list[str]) -> int:
    cfg = load()
    cmd = next((arg for arg in argv if not arg.startswith("-")), "table")
    if cmd == "manifest" and "--detect" not in argv:
        share = resolve(
            cfg,
            memory_mib=int(cfg["hardware"]["target_gpu_memory_mib"]),
            gpu_name=str(cfg["hardware"]["target_gpu"]),
            use_live_driver=False,
        )
    else:
        share = resolve(cfg)
    if cmd == "exports":
        sys.stderr.write(share_comment(share) + "\n")
        sys.stdout.write(export_lines(share))
        return 0
    if cmd == "manifest":
        text = render_manifest(cfg, share)
        write = None
        if "--write" in argv:
            write = argv[argv.index("--write") + 1]
        if write:
            path = Path(write)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path}", file=sys.stderr)
        else:
            sys.stdout.write(text)
        return 0
    print_table(cfg, share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
