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

from aegis_sql.research.selective_resampling import ResamplingThresholds, evaluate_policy


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
    result = evaluate_policy(_rows(args.input), thresholds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"items: {result['items']}")
    print(f"triggered: {result['triggered']} ({result['trigger_rate']:.1%})")
    if "policy_accuracy" in result:
        print(f"baseline accuracy: {result['baseline_accuracy']:.1%}")
        print(f"policy accuracy: {result['policy_accuracy']:.1%}")
        print(f"accuracy delta: {result['accuracy_delta']:+.1%}")
    else:
        print("accuracy: not computed (no observed resampled outcomes in input)")
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
