#!/usr/bin/env python3
"""Summarize four comparable Spider-KO schema-style reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aegis_sql.research.spider_ablation import summarize_schema_ablation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Spider-KO schema representation ablation")
    parser.add_argument("--slm", required=True)
    parser.add_argument("--ddl", required=True)
    parser.add_argument("--compact", required=True)
    parser.add_argument("--mschema", required=True)
    parser.add_argument("--expected-items", type=int, default=1034)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _load(path: str) -> dict:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"report must be a JSON object: {source}")
    return payload


def main() -> int:
    args = parse_args()
    reports = {
        "slm": _load(args.slm),
        "ddl": _load(args.ddl),
        "compact": _load(args.compact),
        "mschema": _load(args.mschema),
    }
    summary = summarize_schema_ablation(reports, expected_items=args.expected_items)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"summary: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
