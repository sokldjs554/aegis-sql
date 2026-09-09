#!/usr/bin/env python3
"""Freeze or materialize the exact synthetic data used by the Qwen experiment."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from aegis_sql.config import PROJECT_ROOT
from aegis_sql.training.hf_experiment import load_jsonl, records_sha256, select_records

SNAPSHOT_NAME = "aegis-qwen-flywheel-v1"
SPLITS = ("train", "dev")


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    body = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    return body.encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_gzip(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw, gzip.GzipFile(
        filename="", mode="wb", fileobj=raw, mtime=0
    ) as compressed:
        compressed.write(payload)


def _generator_hashes() -> dict[str, str]:
    files = (
        "scripts/build_demo_db.py",
        "src/aegis_sql/schema/profile.py",
        "src/aegis_sql/flywheel/sql_sampler.py",
        "src/aegis_sql/flywheel/back_translate.py",
        "src/aegis_sql/flywheel/augment.py",
        "src/aegis_sql/flywheel/quality_filter.py",
        "src/aegis_sql/flywheel/build_dataset.py",
    )
    return {name: _file_sha256(PROJECT_ROOT / name) for name in files}


def freeze(source: Path, out: Path, *, train_limit: int, seed: int) -> dict[str, Any]:
    source_manifest_path = source / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    train = select_records(
        load_jsonl(source / "train.jsonl"),
        limit=train_limit,
        seed=seed,
        shuffle_before_limit=True,
    )
    dev = select_records(
        load_jsonl(source / "dev.jsonl"),
        limit=None,
        seed=seed,
        shuffle_before_limit=False,
    )
    rows_by_split = {"train": train, "dev": dev}

    split_manifest: dict[str, Any] = {}
    for split, rows in rows_by_split.items():
        payload = _jsonl_bytes(rows)
        compressed_path = out / f"{split}.jsonl.gz"
        _write_gzip(compressed_path, payload)
        split_manifest[split] = {
            "count": len(rows),
            "records_sha256": records_sha256(rows),
            "jsonl_sha256": _sha256(payload),
            "gzip_sha256": _file_sha256(compressed_path),
        }

    manifest = {
        "name": SNAPSHOT_NAME,
        "format_version": 1,
        "synthetic_data": True,
        "source": {
            "schema": source_manifest["schema"],
            "config": source_manifest["config"],
            "counts": source_manifest["counts"],
            "leakage": source_manifest["leakage"],
        },
        "database_generation": {
            "scale": 1.0,
            "marketing_consent_rate": 0.40,
            "python_hash_seed": 0,
        },
        "selection": {
            "seed": seed,
            "train": f"seeded shuffle then first {train_limit}",
            "dev": "complete source dev split in file order",
        },
        "splits": split_manifest,
        "generator_files_sha256": _generator_hashes(),
        "comparison_scope": {
            "qwen_base_vs_qlora": "paired on this exact snapshot and KorFin evaluator",
            "historical_aegislm_5_3m": "unpaired until retrained on this snapshot",
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def materialize(snapshot: Path, out: Path) -> dict[str, Any]:
    manifest_path = snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("name") != SNAPSHOT_NAME or manifest.get("format_version") != 1:
        raise ValueError("unsupported Qwen dataset snapshot")

    out.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        compressed_path = snapshot / f"{split}.jsonl.gz"
        expected = manifest["splits"][split]
        if _file_sha256(compressed_path) != expected["gzip_sha256"]:
            raise ValueError(f"{split}: compressed snapshot hash mismatch")
        payload = gzip.decompress(compressed_path.read_bytes())
        if _sha256(payload) != expected["jsonl_sha256"]:
            raise ValueError(f"{split}: JSONL hash mismatch")
        rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]
        if len(rows) != expected["count"] or records_sha256(rows) != expected["records_sha256"]:
            raise ValueError(f"{split}: record count or canonical hash mismatch")
        (out / f"{split}.jsonl").write_bytes(payload)

    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--source", type=Path, required=True)
    freeze_parser.add_argument("--out", type=Path, required=True)
    freeze_parser.add_argument("--train-limit", type=int, default=9000)
    freeze_parser.add_argument("--seed", type=int, default=20260824)

    materialize_parser = sub.add_parser("materialize")
    materialize_parser.add_argument("--snapshot", type=Path, required=True)
    materialize_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "freeze":
        manifest = freeze(args.source, args.out, train_limit=args.train_limit, seed=args.seed)
    else:
        manifest = materialize(args.snapshot, args.out)
    counts = {split: manifest["splits"][split]["count"] for split in SPLITS}
    print(f"{manifest['name']}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
