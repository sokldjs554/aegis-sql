#!/usr/bin/env python3
"""Run one query through the EHRSQL-inspired capability-aware adapter.

Examples:

    python scripts/run_answerability_engine.py "신용점수가 700점 이하인 고객 수는?"
    python scripts/run_answerability_engine.py "지역별 평균 월납보험료를 알려줘"

The default AegisEngine remains unchanged.  This command opts into the narrow
research abstention gate explicitly and prints the resulting envelope as JSON.
"""

from __future__ import annotations

import argparse
import json

from aegis_sql.research.capability_engine import CapabilityAwareEngine
from aegis_sql.types import Tier


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AEGIS with the research answerability gate")
    parser.add_argument("question")
    parser.add_argument("--tier", choices=[tier.value for tier in Tier], default=None)
    parser.add_argument("--no-clarify", action="store_true")
    args = parser.parse_args()

    engine = CapabilityAwareEngine.build()
    try:
        result = engine.ask(
            args.question,
            allow_clarify=not args.no_clarify,
            tier=Tier(args.tier) if args.tier else None,
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str))
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
