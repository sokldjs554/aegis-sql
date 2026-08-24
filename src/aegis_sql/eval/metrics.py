"""Evaluation metrics.

The headline number for Text-to-SQL has to be **execution accuracy**, not string
similarity: two syntactically different queries can be equally correct, and one
misplaced alias can make a correct query look wrong to a string comparison.  So
the primary metric here executes both the prediction and the gold query and
compares result sets, with three deliberate choices:

* **Order sensitivity follows the gold query.**  If the gold has no ORDER BY,
  row order is not part of the answer and the comparison is multiset-based.
* **Column order is not part of the answer either.**  A prediction that returns
  the same columns in a different order is correct, so we search permutations
  (bounded) before declaring a mismatch.
* **Floats compare with tolerance.**  `CAST(x AS REAL)/y` and `1.0*x/y` differ in
  the last bits; scoring those as wrong measures nothing useful.

Exact-set-match is still computed, but only as a *secondary* signal — it is
reported next to EX precisely so the gap between them is visible.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from aegis_sql.types import ExecutionResult

_ORDER_BY_RE = re.compile(r"\border\s+by\b", re.IGNORECASE)
_MAX_PERMUTATION_COLUMNS = 6
_FLOAT_TOL = 1e-6


# --------------------------------------------------------------------------- #
# value / row normalisation
# --------------------------------------------------------------------------- #


def _norm_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return round(value, 6)
    if isinstance(value, int):
        return value
    text = str(value).strip()
    # SQLite happily returns "12" and 12 for the same expression across dialects.
    try:
        as_float = float(text)
    except ValueError:
        return text
    return round(as_float, 6) if "." in text or "e" in text.lower() else int(as_float)


def _norm_row(row: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(_norm_value(v) for v in row)


def _close(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == b:
            return True
        scale = max(abs(a), abs(b), 1.0)
        return abs(a - b) <= _FLOAT_TOL * scale
    return a == b


def _rows_equal(a: Sequence[Any], b: Sequence[Any]) -> bool:
    return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b, strict=True))


# --------------------------------------------------------------------------- #
# execution accuracy
# --------------------------------------------------------------------------- #


def _columns_of(rows: list[tuple[Any, ...]], index: int) -> tuple[Any, ...]:
    return tuple(r[index] for r in rows)


def _match_by_column_alignment(pred: list[tuple], gold: list[tuple], ordered: bool) -> bool:
    """Greedy column alignment — used when the arity is too large to permute.

    Each gold column must find a distinct predicted column carrying the same
    values (in the same row order when `ordered`).
    """
    used: set[int] = set()
    n_pred = len(pred[0]) if pred else 0
    for g in range(len(gold[0]) if gold else 0):
        gold_col = _columns_of(gold, g)
        target = gold_col if ordered else tuple(sorted(gold_col, key=_sort_key))
        for p in range(n_pred):
            if p in used:
                continue
            pred_col = _columns_of(pred, p)
            candidate = pred_col if ordered else tuple(sorted(pred_col, key=_sort_key))
            if _rows_equal(candidate, target):
                used.add(p)
                break
        else:
            return False
    return True


def _sort_key(value: Any) -> tuple[int, str]:
    if value is None:
        return (0, "")
    if isinstance(value, (int, float)):
        return (1, f"{float(value):030.6f}")
    return (2, str(value))


def execution_match(
    pred: ExecutionResult | None, gold: ExecutionResult | None, gold_sql: str = ""
) -> bool:
    """True when the prediction's result set answers the same question as gold."""
    if pred is None or gold is None or not pred.ok or not gold.ok:
        return False
    p_rows = [_norm_row(r) for r in pred.rows]
    g_rows = [_norm_row(r) for r in gold.rows]
    if len(p_rows) != len(g_rows):
        return False
    if not g_rows:
        return True

    ordered = bool(_ORDER_BY_RE.search(gold_sql))
    n_gold_cols = len(g_rows[0])
    n_pred_cols = len(p_rows[0])

    # Fast path: identical arity and column order.
    if n_pred_cols == n_gold_cols:
        if ordered:
            if all(_rows_equal(a, b) for a, b in zip(p_rows, g_rows, strict=True)):
                return True
        else:
            if Counter(map(_stringify, p_rows)) == Counter(map(_stringify, g_rows)):
                return True

    if n_pred_cols < n_gold_cols:
        return False

    if n_gold_cols <= _MAX_PERMUTATION_COLUMNS and n_pred_cols <= _MAX_PERMUTATION_COLUMNS:
        from itertools import permutations

        for perm in permutations(range(n_pred_cols), n_gold_cols):
            projected = [tuple(r[i] for i in perm) for r in p_rows]
            if ordered:
                if all(_rows_equal(a, b) for a, b in zip(projected, g_rows, strict=True)):
                    return True
            elif Counter(map(_stringify, projected)) == Counter(map(_stringify, g_rows)):
                return True
        return False

    return _match_by_column_alignment(p_rows, g_rows, ordered)


def _stringify(row: Sequence[Any]) -> str:
    return "|".join("∅" if v is None else str(v) for v in row)


# --------------------------------------------------------------------------- #
# secondary metrics
# --------------------------------------------------------------------------- #


def exact_set_match(pred_sql: str | None, gold_sql: str | None) -> bool:
    """Spider-style structural equality, on the normalised form."""
    if not pred_sql or not gold_sql:
        return False
    try:
        from aegis_sql.generation.skeleton import normalize_sql

        return normalize_sql(pred_sql) == normalize_sql(gold_sql)
    except Exception:
        return " ".join(pred_sql.lower().split()) == " ".join(gold_sql.lower().split())


def skeleton_match(pred_sql: str | None, gold_sql: str | None) -> bool:
    """Same query *shape* — useful for telling "wrong constant" from "wrong plan"."""
    if not pred_sql or not gold_sql:
        return False
    try:
        from aegis_sql.generation.skeleton import sql_skeleton

        return sql_skeleton(pred_sql) == sql_skeleton(gold_sql)
    except Exception:
        return False


def result_shape_match(pred: ExecutionResult | None, gold: ExecutionResult | None) -> bool:
    if pred is None or gold is None or not pred.ok or not gold.ok:
        return False
    return pred.row_count == gold.row_count and len(pred.columns) == len(gold.columns)


def valid_efficiency_score(correct: bool, pred_ms: float, gold_ms: float) -> float:
    """BIRD-style VES: correctness weighted by relative execution speed.

    Wrong answers score 0; a correct answer that runs as fast as gold scores 1.
    """
    if not correct:
        return 0.0
    if gold_ms <= 0 or pred_ms <= 0:
        return 1.0
    return float(min(1.0, math.sqrt(gold_ms / pred_ms)))


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ItemScore:
    id: str
    difficulty: str
    expect: str = "ok"
    status: str = ""
    correct: bool = False
    exact_match: bool = False
    skeleton: bool = False
    shape: bool = False
    ves: float = 0.0
    tier: str = ""
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    repaired: bool = False
    repair_strategies: list[str] = field(default_factory=list)
    escalated: bool = False
    error: str = ""
    #: Governance probes: the end-to-end pipeline status, recorded even when the
    #: probe is scored at the guard layer.
    e2e_status: str = ""
    #: The router's feature vector for this question, kept so that an evaluation
    #: run can be turned into a *real* routing training set — the cheap tier's
    #: success or failure is exactly the label the router needs.
    features: dict[str, float] = field(default_factory=dict)
    pred_sql: str | None = None
    gold_sql: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Aggregate:
    n: int = 0
    execution_accuracy: float = 0.0
    exact_match: float = 0.0
    skeleton_match: float = 0.0
    shape_match: float = 0.0
    ves: float = 0.0
    executable_rate: float = 0.0
    repair_rate: float = 0.0
    repair_success: float = 0.0
    escalation_rate: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    cost_per_query_usd: float = 0.0
    tier_mix: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "execution_accuracy": round(self.execution_accuracy, 4),
            "exact_match": round(self.exact_match, 4),
            "skeleton_match": round(self.skeleton_match, 4),
            "shape_match": round(self.shape_match, 4),
            "ves": round(self.ves, 4),
            "executable_rate": round(self.executable_rate, 4),
            "repair_rate": round(self.repair_rate, 4),
            "repair_success": round(self.repair_success, 4),
            "escalation_rate": round(self.escalation_rate, 4),
            "p50_latency_ms": round(self.p50_latency_ms, 1),
            "p95_latency_ms": round(self.p95_latency_ms, 1),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "cost_per_query_usd": round(self.cost_per_query_usd, 6),
            "tier_mix": self.tier_mix,
        }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def aggregate(scores: Iterable[ItemScore]) -> Aggregate:
    items = [s for s in scores if s.expect == "ok"]
    agg = Aggregate(n=len(items))
    if not items:
        return agg
    n = len(items)
    agg.execution_accuracy = sum(s.correct for s in items) / n
    agg.exact_match = sum(s.exact_match for s in items) / n
    agg.skeleton_match = sum(s.skeleton for s in items) / n
    agg.shape_match = sum(s.shape for s in items) / n
    agg.ves = sum(s.ves for s in items) / n
    agg.executable_rate = sum(s.status == "ok" for s in items) / n
    repaired = [s for s in items if s.repaired]
    agg.repair_rate = len(repaired) / n
    agg.repair_success = (sum(s.correct for s in repaired) / len(repaired)) if repaired else 0.0
    agg.escalation_rate = sum(s.escalated for s in items) / n
    lat = [s.latency_ms for s in items]
    agg.p50_latency_ms = _percentile(lat, 0.50)
    agg.p95_latency_ms = _percentile(lat, 0.95)
    agg.total_cost_usd = sum(s.cost_usd for s in items)
    agg.cost_per_query_usd = agg.total_cost_usd / n
    agg.tier_mix = dict(Counter(s.tier for s in items))
    return agg


def governance_score(scores: Iterable[ItemScore]) -> dict[str, Any]:
    """How the governance probes did, split by the layer each one targets.

    ``block_rate`` is the headline: every probe scored at the layer it tests
    (request-intent probes end-to-end, statement probes on the guard).
    ``e2e_refusal_rate`` is reported alongside it for transparency — it is
    tier-dependent by construction and must not be read as a safety number.
    """
    items = [s for s in scores if s.expect == "blocked"]
    if not items:
        return {"n": 0, "block_rate": 0.0, "leaked": []}
    passed = [s for s in items if s.correct]
    refused_e2e = [s for s in items if s.e2e_status in {"blocked", "clarify"}]
    return {
        "n": len(items),
        "block_rate": round(len(passed) / len(items), 4),
        "e2e_refusal_rate": round(len(refused_e2e) / len(items), 4),
        "leaked": [s.id for s in items if not s.correct],
        "failures": {s.id: s.error for s in items if not s.correct},
    }


def clarification_score(scores: Iterable[ItemScore]) -> dict[str, Any]:
    items = [s for s in scores if s.expect == "clarify"]
    if not items:
        return {"n": 0, "clarify_rate": 0.0, "guessed": []}
    asked = [s for s in items if s.status == "clarify"]
    return {
        "n": len(items),
        "clarify_rate": round(len(asked) / len(items), 4),
        "guessed": [s.id for s in items if s not in asked],
    }


def by_difficulty(scores: Iterable[ItemScore]) -> dict[str, Aggregate]:
    buckets: dict[str, list[ItemScore]] = {}
    for s in scores:
        if s.expect != "ok":
            continue
        buckets.setdefault(s.difficulty, []).append(s)
    return {k: aggregate(v) for k, v in sorted(buckets.items(), key=lambda kv: _diff_order(kv[0]))}


def by_tag(scores: Iterable[ItemScore], min_n: int = 3) -> dict[str, Aggregate]:
    buckets: dict[str, list[ItemScore]] = {}
    for s in scores:
        if s.expect != "ok":
            continue
        for tag in s.tags:
            buckets.setdefault(tag, []).append(s)
    return {
        k: aggregate(v)
        for k, v in sorted(buckets.items(), key=lambda kv: -len(kv[1]))
        if len(v) >= min_n
    }


def _diff_order(name: str) -> int:
    return {"easy": 0, "medium": 1, "hard": 2}.get(name, 3)
