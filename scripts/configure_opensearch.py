#!/usr/bin/env python3
"""Register the S3 snapshot repo and enable CAGRA/HNSW remote index build."""

from __future__ import annotations

import os
import sys

import requests


def main() -> None:
    opensearch_url = os.environ.get("OPENSEARCH_URL", "http://127.0.0.1:9200").rstrip("/")
    builder_url = os.environ.get(
        "REMOTE_INDEX_BUILDER_URL", "http://remote-index-builder:1025"
    ).rstrip("/")
    bucket = os.environ.get("S3_BUCKET", "").strip()
    if not bucket:
        raise SystemExit("S3_BUCKET must be set")

    repository = os.environ.get("REMOTE_VECTOR_REPOSITORY", "vector-repo").strip() or "vector-repo"
    region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION") or "us-east-2"
    prefix = os.environ.get("S3_PREFIX", "knn-indexes").strip() or "knn-indexes"
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})

    snapshot = {
        "type": "s3",
        "settings": {"bucket": bucket, "base_path": prefix, "region": region},
    }
    endpoint = os.environ.get("S3_ENDPOINT", "").strip()
    if endpoint:
        snapshot["settings"]["endpoint"] = endpoint
        snapshot["settings"]["path_style_access"] = True
        snapshot["settings"]["protocol"] = "http" if endpoint.startswith("http://") else "https"
        snapshot["settings"]["disable_chunked_encoding"] = True

    resp = session.put(f"{opensearch_url}/_snapshot/{repository}", json=snapshot, timeout=60)
    print("snapshot repo:", resp.status_code, resp.text)
    resp.raise_for_status()

    # Default 5s × INITIAL_DELAY_FACTOR=3 ≈ 15s idle per flush. 200ms → ~0.6s first poll.
    poll_interval = os.environ.get("REMOTE_BUILD_POLL_INTERVAL", "200ms").strip() or "200ms"
    settings = {
        "persistent": {
            "knn.remote_index_build.enabled": True,
            "knn.remote_index_build.repository": repository,
            "knn.remote_index_build.service.endpoint": builder_url,
            "knn.remote_index_build.poll.interval": poll_interval,
        }
    }
    resp = session.put(f"{opensearch_url}/_cluster/settings", json=settings, timeout=60)
    print("cluster settings:", resp.status_code, resp.text)
    resp.raise_for_status()
    print("Remote CAGRA_HNSW index build is enabled")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"configure_opensearch failed: {exc}", file=sys.stderr)
        raise
