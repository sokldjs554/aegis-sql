"""Contract tests for one-shot bounded repair on Spider-KO."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

EVALUATOR = Path("scripts/eval_spider_ko_hf.py")


def test_bounded_repair_targets_only_selected_sqlite_errors_and_only_once():
    from aegis_sql.research.spider_ko import bounded_repair_reason

    assert bounded_repair_reason("no such column: singer.age", attempts=0) == "no_such_column"
    assert bounded_repair_reason("no such table: singer", attempts=0) == "no_such_table"
    assert bounded_repair_reason('ambiguous column name: name', attempts=0) == "ambiguous_column"
    assert bounded_repair_reason('near "FROM": syntax error', attempts=0) == "syntax_error"

    assert bounded_repair_reason("database is locked", attempts=0) is None
    assert bounded_repair_reason("interrupted", attempts=0) is None
    assert bounded_repair_reason("no such column: singer.age", attempts=1) is None


def test_bounded_repair_experiment_requires_mschema():
    from aegis_sql.research.spider_ko import validate_bounded_repair_experiment

    validate_bounded_repair_experiment(enabled=True, schema_style="mschema")
    validate_bounded_repair_experiment(enabled=False, schema_style="slm")

    with pytest.raises(ValueError, match="mschema"):
        validate_bounded_repair_experiment(enabled=True, schema_style="ddl")


def test_repair_prompt_contains_only_runtime_evidence_not_gold_sql():
    from aegis_sql.research.spider_ko import build_bounded_repair_prompt

    parameters = inspect.signature(build_bounded_repair_prompt).parameters
    assert not any("gold" in name.lower() for name in parameters)

    prompt = build_bounded_repair_prompt(
        question="가수 이름을 보여줘",
        schema_card="singer(id INTEGER, name TEXT)",
        initial_sql="SELECT singer_name FROM singer;",
        execution_error="no such column: singer_name",
    )

    assert "가수 이름을 보여줘" in prompt
    assert "singer(id INTEGER, name TEXT)" in prompt
    assert "SELECT singer_name FROM singer;" in prompt
    assert "no such column: singer_name" in prompt
    assert "실패" in prompt or "오류" in prompt
    assert "정답 SQL" not in prompt


def test_evaluator_wires_one_shot_repair_and_preserves_initial_final_evidence():
    text = EVALUATOR.read_text(encoding="utf-8")

    assert "--bounded-repair" in text
    assert "validate_bounded_repair_experiment" in text
    assert "bounded_repair_reason" in text
    assert "build_bounded_repair_prompt" in text
    assert '"repair_max_attempts": 1' in text

    for field in (
        '"initial_pred_sql"',
        '"initial_correct"',
        '"initial_pred_execution_ok"',
        '"initial_pred_error"',
        '"repair_attempted"',
        '"repair_reason"',
        '"repaired_sql"',
        '"repair_latency_ms"',
        '"repair_execution_recovered"',
        '"repair_correct"',
        '"initial_execution_accuracy"',
        '"total_generation_latency_ms"',
    ):
        assert field in text


def test_repair_decision_is_independent_of_gold_query_execution():
    text = EVALUATOR.read_text(encoding="utf-8")
    decision = text.index("bounded_repair_reason(initial_error")
    gold_execution = text.index("gold_result = executor.execute(item.gold_sql)")

    assert decision < gold_execution
    decision_block = text[decision:gold_execution]
    assert "gold_result" not in decision_block
