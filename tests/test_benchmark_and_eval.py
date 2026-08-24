"""Benchmark integrity and the metric implementations.

A benchmark is only as trustworthy as its gold labels, so these tests re-execute
every gold query against the live database.  If a gold SQL stops running (schema
change, generator change), the suite fails here rather than silently reporting a
lower accuracy for the engine.
"""

from __future__ import annotations

import pytest

from aegis_sql.eval.metrics import (
    ItemScore,
    aggregate,
    clarification_score,
    exact_set_match,
    execution_match,
    governance_score,
    valid_efficiency_score,
)
from aegis_sql.types import ExecutionResult


def test_benchmark_shape(benchmark):
    assert len(benchmark) >= 100
    kinds = {i.expect for i in benchmark}
    assert kinds == {"ok", "blocked", "clarify"}
    answerable = [i for i in benchmark if i.expect == "ok"]
    assert {i.difficulty for i in answerable} == {"easy", "medium", "hard"}
    assert all(i.gold_sql for i in answerable)
    assert len({i.id for i in benchmark}) == len(benchmark), "duplicate ids"


def test_every_gold_query_executes(benchmark, executor):
    failures = []
    for item in benchmark:
        if item.expect != "ok":
            continue
        result = executor.execute(item.gold_sql)
        if not result.ok:
            failures.append((item.id, result.error))
    assert not failures, f"gold SQL no longer executes: {failures[:5]}"


def test_gold_queries_return_rows(benchmark, executor):
    """An empty gold result makes the item trivially satisfiable — none allowed."""
    empty = [
        item.id
        for item in benchmark
        if item.expect == "ok" and executor.execute(item.gold_sql).row_count == 0
    ]
    assert not empty, f"gold queries with no rows: {empty}"


def test_governance_probes_declare_expected_violation(benchmark):
    probes = [i for i in benchmark if i.expect == "blocked"]
    assert len(probes) >= 6
    assert all(i.expected_violation for i in probes)


def test_execution_match_ignores_column_order():
    gold = ExecutionResult(ok=True, columns=["a", "b"], rows=[(1, "x"), (2, "y")], row_count=2)
    pred = ExecutionResult(ok=True, columns=["b", "a"], rows=[("y", 2), ("x", 1)], row_count=2)
    assert execution_match(pred, gold, "SELECT a, b FROM t")


def test_execution_match_respects_order_by():
    gold = ExecutionResult(ok=True, columns=["a"], rows=[(1,), (2,)], row_count=2)
    pred = ExecutionResult(ok=True, columns=["a"], rows=[(2,), (1,)], row_count=2)
    assert not execution_match(pred, gold, "SELECT a FROM t ORDER BY a")
    assert execution_match(pred, gold, "SELECT a FROM t")


def test_execution_match_float_tolerance():
    a = ExecutionResult(ok=True, columns=["r"], rows=[(1 / 3,)], row_count=1)
    b = ExecutionResult(ok=True, columns=["r"], rows=[(0.33333334,)], row_count=1)
    assert execution_match(a, b, "")


def test_execution_match_rejects_wrong_rows():
    gold = ExecutionResult(ok=True, columns=["a"], rows=[(1,)], row_count=1)
    pred = ExecutionResult(ok=True, columns=["a"], rows=[(2,)], row_count=1)
    assert not execution_match(pred, gold, "")


def test_execution_match_requires_success():
    gold = ExecutionResult(ok=True, columns=["a"], rows=[(1,)], row_count=1)
    assert not execution_match(ExecutionResult(ok=False, error="boom"), gold, "")


def test_exact_set_match_normalises_whitespace():
    assert exact_set_match("SELECT   a  FROM t", "select a from t")


def test_ves_rewards_speed():
    assert valid_efficiency_score(False, 1, 1) == 0.0
    assert valid_efficiency_score(True, 100, 100) == pytest.approx(1.0)
    assert valid_efficiency_score(True, 400, 100) < 1.0


def test_aggregate_excludes_probes():
    scores = [
        ItemScore(id="a", difficulty="easy", correct=True, status="ok"),
        ItemScore(id="b", difficulty="easy", correct=False, status="failed"),
        ItemScore(id="g", difficulty="governance", expect="blocked", status="blocked", correct=True),
    ]
    agg = aggregate(scores)
    assert agg.n == 2
    assert agg.execution_accuracy == pytest.approx(0.5)
    assert governance_score(scores)["block_rate"] == pytest.approx(1.0)
    assert clarification_score(scores)["n"] == 0
