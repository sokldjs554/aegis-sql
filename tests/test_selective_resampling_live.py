from __future__ import annotations

from aegis_sql.research.selective_resampling import (
    evaluate_policy,
    ids_sha256,
    summarize_live_experiment,
)


def _generation(requested: int, completed: int) -> dict:
    return {"generation": {"requested_samples": requested, "completed_samples": completed}}


def _rows() -> list[dict]:
    return [
        {
            "id": "triggered",
            "difficulty": "hard",
            "route_confidence": 0.30,
            "groups": 3,
            "agreement": 0.40,
            "candidate_count": 5,
            "trigger": True,
            "baseline_correct": False,
            "resampled_correct": True,
            "baseline_cost_usd": 0.10,
            "resample_extra_cost_usd": 0.04,
            "baseline_latency_ms": 100.0,
            "resample_extra_latency_ms": 80.0,
            "baseline": {
                "tier": "ensemble",
                "vote_observed": True,
                **_generation(5, 5),
            },
            "resample": {"tier": "ensemble", **_generation(5, 5)},
        },
        {
            "id": "stable",
            "difficulty": "easy",
            "route_confidence": 0.90,
            "groups": 1,
            "agreement": 1.0,
            "candidate_count": 1,
            "trigger": False,
            "baseline_correct": True,
            "resampled_correct": None,
            "baseline_cost_usd": 0.0,
            "resample_extra_cost_usd": 0.0,
            "baseline_latency_ms": 20.0,
            "resample_extra_latency_ms": 0.0,
            "baseline": {
                "tier": "template",
                "vote_observed": False,
                **_generation(1, 1),
            },
            "resample": None,
        },
    ]


def _manifest() -> dict:
    ids = ["triggered", "stable"]
    return {
        "run_kind": "full",
        "provider_requested": "anthropic",
        "provider_actual": "anthropic",
        "model_actual": "claude-sonnet-5",
        "git": {"commit": "a" * 40, "dirty": False},
        "benchmark": {
            "items": 2,
            "sha256": "b" * 64,
            "selected_ids_sha256": ids_sha256(ids),
        },
        "database": {
            "sha256": "c" * 64,
            "schema_fingerprint": "schema-v1",
        },
        "prompt_manifest": {"nl2sql.user": "nl2sql.user@1#abc"},
        "router_artifacts_sha256": {"models/router/router_weights.npz": "d" * 64},
        "runtime": {"pythonhashseed": "0"},
        "thresholds": {
            "confidence": 0.60,
            "agreement": 0.60,
            "min_groups": 2,
            "min_candidates": 2,
        },
        "protocol": {"resample_samples": 5},
    }


def test_live_summary_reports_paired_ex_cost_and_latency():
    report = summarize_live_experiment(_rows(), _manifest(), expected_items=2)
    metrics = report["metrics"]

    assert report["portfolio_evidence_ready"]
    assert metrics["triggered"] == 1
    assert metrics["baseline_correct"] == 1
    assert metrics["policy_correct"] == 2
    assert metrics["accuracy_delta_pp"] == 50.0
    assert metrics["cost_usd"]["extra_total"] == 0.04
    assert metrics["latency_ms"]["baseline_p50"] == 60.0
    assert metrics["latency_ms"]["policy_p95"] == 172.0
    assert metrics["transitions"] == {"gains": 1, "regressions": 0, "unchanged": 1}


def test_missing_triggered_outcome_is_not_coerced_to_incorrect():
    rows = _rows()
    rows[0]["resampled_correct"] = None
    report = evaluate_policy(rows)

    assert report["triggered"] == 1
    assert report["comparable_items"] == 1
    assert report["baseline_accuracy"] == 1.0
    assert report["policy_accuracy"] == 1.0


def test_strict_gate_rejects_mock_or_partial_run():
    manifest = _manifest()
    manifest["run_kind"] = "smoke"
    manifest["provider_requested"] = "mock"
    manifest["provider_actual"] = "mock"
    manifest["model_actual"] = "mock"

    report = summarize_live_experiment(_rows()[:1], manifest, expected_items=2)

    assert not report["portfolio_evidence_ready"]
    assert any("hosted provider" in error for error in report["evidence_errors"])
    assert any("raw row count" in error for error in report["evidence_errors"])
