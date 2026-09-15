"""Comparable summaries for Spider-KO schema-representation ablations."""

from __future__ import annotations

from typing import Any, Mapping

SCHEMA_STYLES = ("slm", "ddl", "compact", "mschema")
_BASELINE_STYLE = "slm"
_REQUIRED_ROW_FIELDS = {
    "db_id",
    "question",
    "gold_sql",
    "pred_sql",
    "correct",
    "pred_execution_ok",
    "pred_error",
    "gold_execution_ok",
    "gold_error",
}


def _require_complete_report(style: str, report: Mapping[str, Any], expected_items: int) -> None:
    if report.get("items") != expected_items:
        raise ValueError(f"{style}: expected {expected_items} items")
    if report.get("portfolio_evidence_ready") is not True:
        raise ValueError(f"{style}: portfolio_evidence_ready must be true")
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != expected_items:
        raise ValueError(f"{style}: row-level evidence must contain {expected_items} items")
    if not all(isinstance(row, dict) and _REQUIRED_ROW_FIELDS.issubset(row) for row in rows):
        raise ValueError(f"{style}: incomplete row-level evidence")
    actual_style = (report.get("evaluation") or {}).get("schema_style")
    if actual_style != style:
        raise ValueError(f"{style}: report schema_style is {actual_style!r}")


def _provenance(report: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = report.get("evaluation") or {}
    runtime = report.get("runtime") or {}
    gpu = runtime.get("gpu") or {}
    packages = runtime.get("packages") or {}
    return {
        "benchmark": report.get("benchmark"),
        "split": report.get("split"),
        "question_language": report.get("question_language"),
        "model": report.get("model"),
        "adapter": report.get("adapter"),
        "git_sha": evaluation.get("git_sha"),
        "dataset_sha256": evaluation.get("dataset_sha256"),
        "database_sha256": evaluation.get("database_sha256"),
        "quantization": evaluation.get("quantization"),
        "max_new_tokens": evaluation.get("max_new_tokens"),
        "warmup_runs": evaluation.get("warmup_runs"),
        "python": runtime.get("python"),
        "torch": runtime.get("torch"),
        "cuda_runtime": runtime.get("cuda_runtime"),
        "transformers": packages.get("transformers"),
        "peft": packages.get("peft"),
        "accelerate": packages.get("accelerate"),
        "bitsandbytes": packages.get("bitsandbytes"),
        "gpu_name": gpu.get("name"),
    }


def _validate_provenance(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    baseline = _provenance(reports[_BASELINE_STYLE])
    for style in SCHEMA_STYLES[1:]:
        current = _provenance(reports[style])
        for key, baseline_value in baseline.items():
            if current.get(key) != baseline_value:
                raise ValueError(f"{style}: provenance mismatch for {key}")
    return baseline


def _style_metrics(report: Mapping[str, Any]) -> dict[str, Any]:
    rows = report["rows"]
    no_such_column = 0
    no_such_table = 0
    pred_execution_failures = 0
    gold_execution_failures = 0
    for row in rows:
        if not row["pred_execution_ok"]:
            pred_execution_failures += 1
        if not row["gold_execution_ok"]:
            gold_execution_failures += 1
        error = str(row.get("pred_error") or "").lower()
        if "no such column" in error:
            no_such_column += 1
        if "no such table" in error:
            no_such_table += 1

    items = int(report["items"])
    correct = int(report["correct"])
    execution_accuracy = correct / items
    execution_failures = sum(
        not row["pred_execution_ok"] or not row["gold_execution_ok"] for row in rows
    )
    recorded_failures = int(report.get("execution_failures", execution_failures))
    if execution_failures != recorded_failures:
        raise ValueError("execution_failures does not match row-level evidence")

    latency = report.get("latency_ms") or {}
    return {
        "items": items,
        "correct": correct,
        "execution_accuracy": execution_accuracy,
        "execution_failures": execution_failures,
        "pred_execution_failures": pred_execution_failures,
        "gold_execution_failures": gold_execution_failures,
        "no_such_column": no_such_column,
        "no_such_table": no_such_table,
        "schema_reference_failures": no_such_column + no_such_table,
        "latency_ms": {
            "p50": latency.get("p50"),
            "p95": latency.get("p95"),
        },
    }


def summarize_schema_ablation(
    reports: Mapping[str, Mapping[str, Any]], *, expected_items: int = 1034
) -> dict[str, Any]:
    """Validate four comparable full reports and summarize the style effect."""
    if expected_items < 1:
        raise ValueError("expected_items must be positive")

    missing = [style for style in SCHEMA_STYLES if style not in reports]
    extra = [style for style in reports if style not in SCHEMA_STYLES]
    if missing or extra:
        detail = []
        if missing:
            detail.append(f"missing: {', '.join(missing)}")
        if extra:
            detail.append(f"unexpected: {', '.join(extra)}")
        raise ValueError("schema styles must be exactly slm, ddl, compact, mschema; " + "; ".join(detail))

    for style in SCHEMA_STYLES:
        _require_complete_report(style, reports[style], expected_items)
    provenance = _validate_provenance(reports)

    styles = {style: _style_metrics(reports[style]) for style in SCHEMA_STYLES}
    baseline = styles[_BASELINE_STYLE]
    for metrics in styles.values():
        metrics["delta_accuracy_pp_vs_slm"] = round(
            (metrics["execution_accuracy"] - baseline["execution_accuracy"]) * 100,
            2,
        )
        metrics["delta_execution_failures_vs_slm"] = (
            metrics["execution_failures"] - baseline["execution_failures"]
        )
        metrics["delta_schema_reference_failures_vs_slm"] = (
            metrics["schema_reference_failures"] - baseline["schema_reference_failures"]
        )

    winner = min(
        SCHEMA_STYLES,
        key=lambda style: (
            -styles[style]["execution_accuracy"],
            styles[style]["schema_reference_failures"],
            styles[style]["execution_failures"],
            float(styles[style]["latency_ms"].get("p95") or 0.0),
        ),
    )
    return {
        "comparable": True,
        "baseline_style": _BASELINE_STYLE,
        "winner": winner,
        "expected_items": expected_items,
        "provenance": provenance,
        "styles": styles,
    }


__all__ = ["SCHEMA_STYLES", "summarize_schema_ablation"]
