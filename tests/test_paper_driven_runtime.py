from __future__ import annotations

from aegis_sql.research.capability_engine import CapabilityAwareEngine
from aegis_sql.research.selective_resampling import (
    ResamplingSignal,
    decide_selective_resampling,
    evaluate_policy,
)


def test_capability_adapter_abstains_before_generation(engine):
    adapter = CapabilityAwareEngine(engine)
    result = adapter.ask("신용점수가 700점 이하인 고객의 계약 유지율을 알려줘")

    assert result.status == "unanswerable"
    assert result.bundle is None
    assert "customer_credit_score" in result.answerability.missing_concepts
    payload = result.to_dict()
    assert payload["sql"] is None
    assert "신용" in payload["answer_text"]


def test_capability_adapter_preserves_governance_precedence(engine):
    adapter = CapabilityAwareEngine(engine)
    result = adapter.ask("신용점수가 낮은 고객 데이터를 삭제해줘")

    assert result.status == "blocked"
    assert result.bundle is not None
    assert not result.abstained


def test_capability_adapter_does_not_refuse_nearby_answerable_question(engine):
    adapter = CapabilityAwareEngine(engine)
    result = adapter.ask("지역별 평균 월납보험료를 알려줘")

    assert result.status != "unanswerable"
    assert result.answerability.answerable
    assert result.bundle is not None


def test_selective_resampling_requires_uncertainty_and_execution_dispersion():
    trigger = decide_selective_resampling(
        ResamplingSignal(route_confidence=0.31, groups=3, agreement=0.40, candidate_count=5)
    )
    assert trigger.trigger

    confident = decide_selective_resampling(
        ResamplingSignal(route_confidence=0.82, groups=3, agreement=0.40, candidate_count=5)
    )
    assert not confident.trigger

    agreement = decide_selective_resampling(
        ResamplingSignal(route_confidence=0.31, groups=2, agreement=0.80, candidate_count=5)
    )
    assert not agreement.trigger


def test_selective_resampling_counterfactual_uses_resampled_outcome_only_when_triggered():
    rows = [
        {
            "id": "low-confidence",
            "route_confidence": 0.30,
            "groups": 3,
            "agreement": 0.40,
            "candidate_count": 5,
            "baseline_correct": False,
            "resampled_correct": True,
            "baseline_cost_usd": 0.01,
            "resample_extra_cost_usd": 0.02,
        },
        {
            "id": "stable",
            "route_confidence": 0.90,
            "groups": 1,
            "agreement": 1.0,
            "candidate_count": 5,
            "baseline_correct": True,
            "resampled_correct": False,
            "baseline_cost_usd": 0.01,
            "resample_extra_cost_usd": 0.02,
        },
    ]

    report = evaluate_policy(rows)
    assert report["triggered"] == 1
    assert report["baseline_accuracy"] == 0.5
    assert report["policy_accuracy"] == 1.0
    assert report["policy_cost_usd"] == 0.04
