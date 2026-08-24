#!/usr/bin/env python3
"""Search for a better prompt, scored by execution accuracy on a dev split.

    python scripts/optimize_prompts.py --prompt nl2sql.user --generations 3

Every variant is evaluated with the real engine against held-out benchmark items,
so the reported delta is an accuracy delta, not an impression.  The winning
template is written to ``reports/prompt_opt.json`` together with the full search
history; promoting it is a deliberate, reviewable edit to
``configs/prompts/default.yaml`` — the optimiser never rewrites the registry
behind your back.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aegis_sql.config import get_settings  # noqa: E402
from aegis_sql.observability.logging import configure_logging  # noqa: E402
from aegis_sql.prompts.optimizer import (  # noqa: E402
    PromptOptimizer,
    benchmark_evaluator,
    save_result,
)
from aegis_sql.prompts.registry import PromptRegistry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Automatic prompt optimisation")
    ap.add_argument("--prompt", default="nl2sql.user", help="prompt id to optimise")
    ap.add_argument("--set", dest="prompt_set", default="default")
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--population", type=int, default=4)
    ap.add_argument("--dev-limit", type=int, default=24)
    ap.add_argument("--out", default="reports/prompt_opt.json")
    args = ap.parse_args()

    configure_logging("INFO")
    settings = get_settings()
    registry = PromptRegistry.load(args.prompt_set)

    if settings.generation.provider == "template":
        print(
            "[warn] provider=template — 프롬프트는 이 티어에서 사용되지 않으므로 델타가 0으로 나옵니다.\n"
            "       ANTHROPIC_API_KEY 또는 OPENAI_API_KEY 를 설정한 뒤 다시 실행하세요.",
            file=sys.stderr,
        )

    llm = None
    try:
        from aegis_sql.llm.providers import get_llm_client

        client = get_llm_client(settings)
        llm = client if client.available() else None
    except Exception:
        pass

    optimizer = PromptOptimizer(
        registry=registry,
        prompt_id=args.prompt,
        evaluate=benchmark_evaluator(settings, dev_limit=args.dev_limit),
        settings=settings,
        llm=llm,
    )
    result = optimizer.optimize(generations=args.generations, population=args.population)

    print(f"\nbaseline EX : {result.baseline_score:.1%}")
    print(f"best     EX : {result.best_score:.1%}  ({result.delta * 100:+.1f}%p)")
    print(f"operator    : {result.best_operator}")
    print(f"evaluated   : {result.evaluated} variants in {result.wall_s:.0f}s\n")
    for row in result.history:
        print(f"  gen{row['generation']}  {row['score']:.3f}  {row['tokens']:>5}tok  "
              f"{row['operator']:<22} {row['name'][:40]}")
    path = save_result(result, args.out)
    print(f"\n→ {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
