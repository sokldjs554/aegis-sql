"""Public model evidence must stay tied to the archived measurements."""

from __future__ import annotations

from aegis_sql.config import PROJECT_ROOT
from aegis_sql.research.evidence import (
    build_model_experiment_evidence,
    load_published_model_experiment_evidence,
)


def test_published_model_evidence_matches_primary_artifacts():
    assert load_published_model_experiment_evidence() == build_model_experiment_evidence()


def test_model_evidence_keeps_live_and_recorded_scopes_separate():
    evidence = build_model_experiment_evidence()

    assert evidence["public_demo"]["live_tier"] == "template"
    assert evidence["public_demo"]["recorded_results_are_live"] is False

    runs = {run["key"]: run for run in evidence["tier_comparison"]["runs"]}
    assert runs["template"]["correct"] == 40
    assert runs["cascade"]["correct"] == 47
    assert runs["llm_only"]["correct"] == 52
    assert runs["template"]["by_difficulty"]["hard"] == 0.0
    assert runs["llm_only"]["by_difficulty"]["hard"] == 0.3
    assert "서로 다른 실행 환경" in evidence["tier_comparison"]["comparison_scope_ko"]

    replays = evidence["tier_comparison"]["replays"]
    assert [case["difficulty"] for case in replays] == ["easy", "medium", "hard"]
    assert [case["id"] for case in replays] == ["kfb-e08", "kfb-m24", "kfb-h17"]
    assert all(case["template"]["correct"] is False for case in replays)
    assert all(case["llm"]["correct"] is True for case in replays)
    assert all(case["gold"]["columns"] for case in replays)
    assert all(case["gold"]["preview"] for case in replays)


def test_model_evidence_discloses_both_evidence_boundaries():
    evidence = build_model_experiment_evidence()
    qwen = evidence["qwen"]
    r3 = evidence["selective_resampling"]

    assert qwen["base"]["correct"] == 10
    assert qwen["adapted"]["correct"] == 11
    assert qwen["evidence"]["raw_result_bundle_archived"] is False
    assert qwen["evidence"]["predicted_sql_archived"] is False

    assert r3["triggered"] == 4
    assert r3["baseline"]["correct"] == 47
    assert r3["policy"]["correct"] == 46
    assert r3["decision"] == "rejected"
    assert "Python 3.13.15" in r3["run_scope_ko"]
    assert "few-shot 0" in r3["run_scope_ko"]
    assert r3["evidence"]["raw_rows_archived"] == 90
    assert r3["evidence"]["row_level_bundle_archived"] is True

    source_paths = {source["path"] for source in evidence["sources"]}
    assert all((PROJECT_ROOT / path).is_file() for path in source_paths)
    assert all(len(source["sha256"]) == 64 for source in evidence["sources"])
