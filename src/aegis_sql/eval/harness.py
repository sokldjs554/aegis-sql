"""The evaluation harness.

A number without a reproducible procedure behind it is decoration, so this
harness fixes every free variable: the database is deterministic, the benchmark's
gold SQL is executed at build time, the reference date is pinned, and the report
records the prompt hashes, schema fingerprint and engine settings that produced
the run.  Re-running it on the same commit produces the same table.

It also runs **ablations** — the same benchmark against an engine with one
component disabled — because "our system scores X" is far less informative than
"removing the business glossary costs Y points on medium queries".
"""

from __future__ import annotations

import json
import platform
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aegis_sql.config import PROJECT_ROOT, Settings, get_settings
from aegis_sql.eval.metrics import (
    Aggregate,
    ItemScore,
    aggregate,
    by_difficulty,
    by_tag,
    clarification_score,
    exact_set_match,
    execution_match,
    governance_score,
    result_shape_match,
    skeleton_match,
    valid_efficiency_score,
)
from aegis_sql.observability.logging import get_logger
from aegis_sql.pipeline import AegisEngine
from aegis_sql.types import AnswerStatus, Tier

log = get_logger("eval.harness")


# --------------------------------------------------------------------------- #
# benchmark loading
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class BenchItem:
    id: str
    question: str
    gold_sql: str | None
    difficulty: str
    expect: str = "ok"
    tables: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    expected_violation: str | None = None
    note: str = ""


def load_benchmark(path: str | Path) -> list[BenchItem]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"benchmark not found: {p}\n  run `make benchmark` first.")
    items: list[BenchItem] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        items.append(
            BenchItem(
                id=raw["id"], question=raw["question"], gold_sql=raw.get("gold_sql"),
                difficulty=raw["difficulty"], expect=raw.get("expect", "ok"),
                tables=raw.get("tables", []), tags=raw.get("tags", []),
                expected_violation=raw.get("expected_violation"), note=raw.get("note", ""),
            )
        )
    return items


# --------------------------------------------------------------------------- #
# ablations
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Variant:
    """One engine configuration to score.

    ``overrides`` are merged into :class:`Settings`; ``mutate`` gets the built
    engine and can disable a component that has no settings switch (e.g. empty
    the glossary) — that keeps the ablation honest instead of approximating it.
    """

    name: str
    description: str
    overrides: dict[str, Any] = field(default_factory=dict)
    mutate: Callable[[AegisEngine], None] | None = None


def _drop_glossary(engine: AegisEngine) -> None:
    engine.c.glossary.entries = []
    engine.c.linker.glossary.entries = []
    engine.c.linker.build_index()


def _full_schema(engine: AegisEngine) -> None:
    """No pruning: hand the generator every table, as a naive system would."""
    from aegis_sql.types import LinkedSchema

    schema = engine.c.schema
    all_tables = list(schema.tables)
    all_columns = [c.qualified for c in schema.all_columns]
    original = engine.c.linker.link

    def link_everything(nq):
        linked: LinkedSchema = original(nq)
        linked.tables = all_tables
        linked.columns = all_columns
        linked.coverage = 1.0
        return linked

    engine.c.linker.link = link_everything  # type: ignore[method-assign]


DEFAULT_ABLATIONS: list[Variant] = [
    Variant("full", "전체 구성 (기준선)"),
    Variant("no-glossary", "사내 용어사전 제거", mutate=_drop_glossary),
    Variant("no-schema-linking", "스키마 프루닝 없이 전체 스키마 투입", mutate=_full_schema),
    Variant("no-value-link", "프로파일된 값 매칭 제거",
            {"retrieval": {"value_match_weight": 0.0}}),
    Variant("dense-only", "BM25 제거 (임베딩 유사도만)",
            {"retrieval": {"dense_weight": 1.0}}),
    Variant("lexical-only", "임베딩 제거 (BM25만)",
            {"retrieval": {"dense_weight": 0.0}}),
    Variant("no-repair", "실행 기반 자가교정 제거",
            {"verify": {"max_repair_attempts": 0}}),
    Variant("card-compact", "스키마 카드를 compact 형식으로",
            {"generation": {"schema_card_style": "compact"}}),
]


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class RunResult:
    variant: str
    description: str
    scores: list[ItemScore]
    overall: Aggregate
    per_difficulty: dict[str, Aggregate]
    per_tag: dict[str, Aggregate]
    governance: dict[str, Any]
    clarification: dict[str, Any]
    wall_s: float
    settings_digest: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "description": self.description,
            "overall": self.overall.as_dict(),
            "per_difficulty": {k: v.as_dict() for k, v in self.per_difficulty.items()},
            "per_tag": {k: v.as_dict() for k, v in self.per_tag.items()},
            "governance": self.governance,
            "clarification": self.clarification,
            "wall_s": round(self.wall_s, 2),
            "settings": self.settings_digest,
            "items": [
                {
                    "id": s.id, "difficulty": s.difficulty, "expect": s.expect, "status": s.status,
                    "correct": s.correct, "exact_match": s.exact_match, "tier": s.tier,
                    "latency_ms": round(s.latency_ms, 1), "cost_usd": round(s.cost_usd, 6),
                    "repaired": s.repaired, "repair_strategies": s.repair_strategies,
                    "error": s.error[:200], "pred_sql": s.pred_sql,
                }
                for s in self.scores
            ],
        }


class EvalHarness:
    def __init__(self, settings: Settings | None = None, bench_path: str | Path | None = None) -> None:
        self.settings = settings or get_settings()
        self.bench_path = bench_path or (PROJECT_ROOT / "data" / "benchmark" / "korfin_bench.jsonl")
        self.items = load_benchmark(self.bench_path)

    # -- selection -------------------------------------------------------- #

    def select(
        self,
        limit: int | None = None,
        difficulties: Iterable[str] | None = None,
        include_probes: bool = True,
    ) -> list[BenchItem]:
        items = self.items
        if difficulties:
            wanted = set(difficulties)
            items = [i for i in items if i.difficulty in wanted or (include_probes and i.expect != "ok")]
        if not include_probes:
            items = [i for i in items if i.expect == "ok"]
        if limit:
            answerable = [i for i in items if i.expect == "ok"][:limit]
            probes = [i for i in items if i.expect != "ok"] if include_probes else []
            items = answerable + probes
        return items

    # -- one variant ------------------------------------------------------ #

    def run_variant(
        self,
        variant: Variant,
        items: list[BenchItem] | None = None,
        tier: Tier | None = None,
        progress: bool = True,
    ) -> RunResult:
        items = items if items is not None else self.select()
        overrides = variant.overrides or {}
        engine = AegisEngine.build(Settings.model_validate({**self.settings.model_dump(), **overrides}))
        if variant.mutate:
            variant.mutate(engine)

        gold_executor = engine.c.executor
        scores: list[ItemScore] = []
        started = time.perf_counter()

        for n, item in enumerate(items, 1):
            score = self._score_item(engine, gold_executor, item, tier)
            scores.append(score)
            if progress and (n % 10 == 0 or n == len(items)):
                acc = sum(s.correct for s in scores if s.expect == "ok")
                den = max(1, sum(1 for s in scores if s.expect == "ok"))
                print(f"    [{variant.name}] {n}/{len(items)}  EX={acc / den:.1%}", flush=True)

        wall = time.perf_counter() - started
        engine.close()
        return RunResult(
            variant=variant.name,
            description=variant.description,
            scores=scores,
            overall=aggregate(scores),
            per_difficulty=by_difficulty(scores),
            per_tag=by_tag(scores),
            governance=governance_score(scores),
            clarification=clarification_score(scores),
            wall_s=wall,
            settings_digest=self._digest(engine, overrides),
        )

    def _score_item(self, engine: AegisEngine, gold_executor, item: BenchItem, tier) -> ItemScore:
        score = ItemScore(
            id=item.id, difficulty=item.difficulty, expect=item.expect,
            tags=item.tags, gold_sql=item.gold_sql,
        )
        # Governance probes must be refused; asking them with clarification enabled
        # would let the engine dodge the test by asking a question instead.
        allow_clarify = item.expect != "blocked"
        bundle = engine.ask(item.question, allow_clarify=allow_clarify, tier=tier)

        score.status = bundle.status.value
        score.tier = bundle.route.tier.value if bundle.route else ""
        score.latency_ms = bundle.total_latency_ms
        score.cost_usd = bundle.cost_usd
        score.repaired = bool(bundle.repairs)
        score.repair_strategies = [s.strategy for s in bundle.repairs]
        score.escalated = bool(bundle.route and bundle.route.escalated_from)
        score.pred_sql = bundle.sql

        if item.expect != "ok":
            if item.expect == "blocked":
                score.correct = bundle.status is AnswerStatus.BLOCKED
                if not score.correct:
                    score.error = f"not blocked (status={bundle.status.value})"
            else:  # clarify
                score.correct = bundle.status is AnswerStatus.CLARIFY
                if not score.correct:
                    score.error = f"answered instead of clarifying (status={bundle.status.value})"
            return score

        if bundle.status is not AnswerStatus.OK or bundle.result is None or not bundle.result.ok:
            score.error = (bundle.result.error if bundle.result else "") or bundle.answer_text
            return score

        gold = gold_executor.execute(item.gold_sql or "")
        score.correct = execution_match(bundle.result, gold, item.gold_sql or "")
        score.exact_match = exact_set_match(bundle.sql, item.gold_sql)
        score.skeleton = skeleton_match(bundle.sql, item.gold_sql)
        score.shape = result_shape_match(bundle.result, gold)
        score.ves = valid_efficiency_score(score.correct, bundle.result.elapsed_ms, gold.elapsed_ms)
        if not score.correct:
            score.error = (
                f"result mismatch (pred {bundle.result.row_count}행 × {len(bundle.result.columns)}열"
                f" vs gold {gold.row_count}행 × {len(gold.columns)}열)"
            )
        return score

    # -- ablation matrix -------------------------------------------------- #

    def run_ablation(
        self,
        variants: list[Variant] | None = None,
        items: list[BenchItem] | None = None,
        progress: bool = True,
    ) -> list[RunResult]:
        variants = variants or DEFAULT_ABLATIONS
        items = items if items is not None else self.select()
        results: list[RunResult] = []
        for v in variants:
            if progress:
                print(f"  ▶ {v.name}: {v.description}", flush=True)
            try:
                results.append(self.run_variant(v, items=items, progress=progress))
            except Exception as exc:  # a broken variant must not kill the matrix
                log.error("ablation variant failed", variant=v.name, error=str(exc), exc_info=True)
        return results

    # -- metadata --------------------------------------------------------- #

    def _digest(self, engine: AegisEngine, overrides: dict[str, Any]) -> dict[str, Any]:
        st = engine.settings
        return {
            "schema_fingerprint": engine.c.schema.fingerprint(),
            "prompt_manifest": engine.c.prompt_registry.manifest() if engine.c.prompt_registry else {},
            "provider": st.generation.provider,
            "model": st.generation.model,
            "available_tiers": sorted(t.value for t, g in engine.c.generators.items() if g.available()),
            "embedder": type(
                getattr(engine.c.linker, "embedder", None)
            ).__name__,
            "retrieval": st.retrieval.model_dump(),
            "verify": st.verify.model_dump(),
            "router": st.router.model_dump(),
            "overrides": overrides,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "benchmark": str(self.bench_path),
            "benchmark_items": len(self.items),
        }
