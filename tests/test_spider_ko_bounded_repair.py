"""Contract tests for one-shot bounded repair on Spider-KO."""

from __future__ import annotations

import inspect

import pytest


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
