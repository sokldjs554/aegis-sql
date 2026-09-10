#!/usr/bin/env python3
"""Validate and summarize one measured Qwen base-versus-QLoRA run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aegis_sql.training.hf_experiment import build_qwen_summary


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a strict Qwen GPU experiment evidence record")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-kind", choices=("smoke", "full"), required=True)
    parser.add_argument("--expected-items", type=int, required=True)
    args = parser.parse_args()

    summary = build_qwen_summary(
        _load(args.manifest),
        _load(args.base),
        _load(args.adapted),
        expected_items=args.expected_items,
        run_kind=args.run_kind,
    )
    summary["artifacts"] = {
        "manifest": str(args.manifest),
        "base_report": str(args.base),
        "adapted_report": str(args.adapted),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    gpu = summary["runtime"]["gpu"]
    print(f"run: {summary['run_kind']} / items={summary['items']} / GPU={gpu['name']}")
    print(
        "training: "
        f"{summary['training']['wall_clock_seconds']:.1f}s / "
        f"peak={summary['training']['peak_cuda_memory_gib']:.3f} GiB"
    )
    for label in ("base", "adapted"):
        block = summary[label]
        difficulty = {k: f"{v['ex']:.1%}" for k, v in block["by_difficulty"].items()}
        print(
            f"{label}: EX={block['execution_accuracy']:.1%} / "
            f"difficulty={difficulty} / latency_ms={block['latency_ms']} / "
            f"peak={block['peak_cuda_memory_gib']:.3f} GiB"
        )
    print(f"delta EX: {summary['delta_ex_percentage_points']:+.2f} percentage points")
    print(f"portfolio evidence ready: {summary['portfolio_evidence_ready']}")
    print(f"summary: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
