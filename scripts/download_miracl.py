#!/usr/bin/env python3
"""Download MIRACL English corpus shards and write the first 1M raw-text docs."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm

REPO_ID = "miracl/miracl-corpus"
PREFIX = "miracl-corpus-v1.0-en"
MAX_SHARDS = 66  # docs-0.jsonl.gz ... docs-65.jsonl.gz


def doc_text(record: dict) -> str:
    title = (record.get("title") or "").strip()
    text = (record.get("text") or "").strip()
    if title and text:
        return f"{title}\n{text}"
    return title or text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1_000_000)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "miracl-en-1m",
    )
    args = parser.parse_args()

    raw_dir = args.out_dir.parent / PREFIX
    raw_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    docs_path = args.out_dir / "docs.jsonl"
    texts_path = args.out_dir / "texts.jsonl"
    meta_path = args.out_dir / "manifest.json"

    written = 0
    shards_used: list[str] = []
    progress = tqdm(total=args.limit, unit="docs", desc="MIRACL en")

    with docs_path.open("w", encoding="utf-8") as docs_out, texts_path.open(
        "w", encoding="utf-8"
    ) as texts_out:
        for shard in range(MAX_SHARDS):
            if written >= args.limit:
                break
            filename = f"{PREFIX}/docs-{shard}.jsonl.gz"
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=filename,
                local_dir=str(raw_dir.parent),
            )
            shards_used.append(filename)
            with gzip.open(local_path, "rt", encoding="utf-8") as src:
                for line in src:
                    if written >= args.limit:
                        break
                    record = json.loads(line)
                    text = doc_text(record)
                    if not text:
                        continue
                    docs_out.write(
                        json.dumps(
                            {
                                "docid": record.get("docid"),
                                "title": record.get("title"),
                                "text": record.get("text"),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    texts_out.write(
                        json.dumps(
                            {"docid": record.get("docid"), "text": text},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    written += 1
                    progress.update(1)

    progress.close()
    meta_path.write_text(
        json.dumps(
            {
                "dataset": REPO_ID,
                "subset": PREFIX,
                "documents": written,
                "shards": shards_used,
                "docs_jsonl": str(docs_path),
                "texts_jsonl": str(texts_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {written:,} documents to {args.out_dir}")


if __name__ == "__main__":
    main()
