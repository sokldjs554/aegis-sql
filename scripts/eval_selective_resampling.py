#!/usr/bin/env python3
"""Evaluate the R³-SQL-inspired selective-resampling gate on recorded pools.

Input is JSONL. Required fields per row:

    id, route_confidence, groups, agreement, candidate_count

Optional counterfactual fields:

    baseline_correct, resampled_correct,
    baseline_cost_usd, resample_extra_cost_usd,
    baseline_latency_ms, resample_extra_latency_ms

No accuracy improvement is claimed unless the input contains *actually observed*
resampled outcomes.  Synthetic placeholders are useful for unit tests only and
must not be used as portfolio evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aegis_sql.research.selective_resampling import (
    ResamplingThresholds,
    evaluate_policy,
    summarize_live_experiment,
)


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate selective resampling on recorded candidate pools")
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, default=Path("reports/selective_resampling.json"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--expected-items", type=int, default=90)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail unless a manifest-backed full hosted run passes the evidence gate",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.60)
    parser.add_argument("--agreement-threshold", type=float, default=0.60)
    parser.add_argument("--min-groups", type=int, default=2)
    parser.add_argument("--min-candidates", type=int, default=2)
    args = parser.parse_args()

    thresholds = ResamplingThresholds(
        confidence=args.confidence_threshold,
        agreement=args.agreement_threshold,
        min_groups=args.min_groups,
        min_candidates=args.min_candidates,
    )
    rows = _rows(args.input)
    if args.manifest is not None:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        report = summarize_live_experiment(
            rows, manifest, thresholds, expected_items=args.expected_items
        )
        result = report["metrics"]
    else:
        if args.strict:
            parser.error("--strict requires --manifest")
        report = evaluate_policy(rows, thresholds)
        result = report
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"items: {result['items']}")
    print(f"triggered: {result['triggered']} ({result['trigger_rate']:.1%})")
    if "policy_accuracy" in result:
        print(f"baseline accuracy: {result['baseline_accuracy']:.1%}")
        print(f"policy accuracy: {result['policy_accuracy']:.1%}")
        print(f"accuracy delta: {result['accuracy_delta_pp']:+.2f} percentage points")
    else:
        print("accuracy: not computed (no observed resampled outcomes in input)")
    cost = result["cost_usd"]
    latency = result["latency_ms"]
    print(f"extra cost: ${cost['extra_total']:.6f}")
    print(
        "latency p50/p95: "
        f"baseline={latency['baseline_p50']:.2f}/{latency['baseline_p95']:.2f} ms, "
        f"policy={latency['policy_p50']:.2f}/{latency['policy_p95']:.2f} ms"
    )
    if args.manifest is not None:
        print(f"portfolio evidence ready: {report['portfolio_evidence_ready']}")
        for error in report["evidence_errors"]:
            print(f"  - {error}")
    print(f"report: {args.out}")
    return 2 if args.strict and not report["portfolio_evidence_ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
