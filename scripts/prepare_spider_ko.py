#!/usr/bin/env python3
"""Export Spider-KO validation to a local JSONL file with provenance metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DATASET_ID = "huggingface-KREW/spider-ko"
SPLIT = "validation"
EXPECTED_ITEMS = 1034


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export the external Spider-KO validation split")
    ap.add_argument("--out", default="data/external/spider-ko-validation.jsonl")
    ap.add_argument("--manifest", default="data/external/spider-ko-validation.manifest.json")
    ap.add_argument("--revision", default="main", help="Hugging Face dataset revision")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit('datasets is missing; run `pip install -e ".[hf]"`') from exc

    dataset = load_dataset(DATASET_ID, split=SPLIT, revision=args.revision)
    if len(dataset) != EXPECTED_ITEMS:
        raise SystemExit(
            f"expected {EXPECTED_ITEMS} Spider-KO validation rows, got {len(dataset)}; "
            "pin/review the dataset revision before treating the run as comparable"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    required = ("db_id", "query", "question", "question_ko")
    with out.open("w", encoding="utf-8") as fh:
        for index, row in enumerate(dataset, 1):
            missing = [key for key in required if not str(row.get(key) or "").strip()]
            if missing:
                raise SystemExit(f"row {index} missing required fields: {', '.join(missing)}")
            payload = {key: row.get(key) for key in row.keys()}
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    manifest = {
        "external_benchmark": True,
        "dataset": DATASET_ID,
        "split": SPLIT,
        "question_language": "ko",
        "revision": args.revision,
        "items": len(dataset),
        "export": str(out),
        "export_sha256": sha256(out),
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
