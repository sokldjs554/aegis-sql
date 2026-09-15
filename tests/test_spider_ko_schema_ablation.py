"""Contract tests for Spider-KO schema-representation ablation."""

from __future__ import annotations

import copy

import pytest

STYLES = ("slm", "ddl", "compact", "mschema")


def _report(style: str, *, correct: int, errors: list[str]) -> dict:
    rows = []
    for index in range(4):
        error = errors[index] if index < len(errors) else ""
        pred_ok = not error
        rows.append(
            {
                "db_id": "school",
                "question": f"질문 {index}",
                "gold_sql": "SELECT 1",
                "pred_sql": "SELECT 1" if pred_ok else "SELECT missing FROM school",
                "correct": index < correct,
                "pred_execution_ok": pred_ok,
                "pred_error": error,
                "gold_execution_ok": True,
                "gold_error": "",
            }
        )
    return {
        "external_benchmark": True,
        "benchmark": "huggingface-KREW/spider-ko",
        "split": "validation",
        "question_language": "ko",
        "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "adapter": None,
        "items": 4,
        "correct": correct,
        "execution_accuracy": correct / 4,
        "execution_failures": sum(not row["pred_execution_ok"] for row in rows),
        "latency_ms": {"p50": 100.0, "p95": 200.0},
        "evaluation": {
            "dataset_sha256": "dataset-sha",
            "database_sha256": {"school": "db-sha"},
            "schema_style": style,
            "quantization": "NF4 4-bit",
        },
        "rows": rows,
        "portfolio_evidence_ready": True,
    }


def test_summary_counts_schema_reference_failures_and_selects_best_style():
    from aegis_sql.research.spider_ablation import summarize_schema_ablation

    reports = {
        "slm": _report("slm", correct=1, errors=["no such column: x", "no such table: y"]),
        "ddl": _report("ddl", correct=2, errors=["no such column: x"]),
        "compact": _report("compact", correct=3, errors=[]),
        "mschema": _report("mschema", correct=2, errors=["misuse of aggregate: COUNT()"]),
    }

    summary = summarize_schema_ablation(reports, expected_items=4)

    assert summary["comparable"] is True
    assert summary["baseline_style"] == "slm"
    assert summary["winner"] == "compact"
    assert summary["styles"]["slm"]["schema_reference_failures"] == 2
    assert summary["styles"]["slm"]["no_such_column"] == 1
    assert summary["styles"]["slm"]["no_such_table"] == 1
    assert summary["styles"]["compact"]["execution_accuracy"] == 0.75
    assert summary["styles"]["compact"]["delta_accuracy_pp_vs_slm"] == 50.0
    assert summary["styles"]["compact"]["delta_schema_reference_failures_vs_slm"] == -2


def test_summary_rejects_provenance_mismatch():
    from aegis_sql.research.spider_ablation import summarize_schema_ablation

    reports = {style: _report(style, correct=2, errors=[]) for style in STYLES}
    reports["ddl"] = copy.deepcopy(reports["ddl"])
    reports["ddl"]["evaluation"]["dataset_sha256"] = "different-dataset"

    with pytest.raises(ValueError, match="dataset_sha256"):
        summarize_schema_ablation(reports, expected_items=4)


def test_summary_requires_all_four_schema_styles():
    from aegis_sql.research.spider_ablation import summarize_schema_ablation

    reports = {style: _report(style, correct=2, errors=[]) for style in STYLES[:-1]}

    with pytest.raises(ValueError, match="mschema"):
        summarize_schema_ablation(reports, expected_items=4)
