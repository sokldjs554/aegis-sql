"""Cascade routing: features, the Keras→numpy round trip, and calibration."""

from __future__ import annotations

import numpy as np
import pytest

from aegis_sql.types import DifficultyFeatures, Tier


def _features(**kw) -> DifficultyFeatures:
    return DifficultyFeatures(**kw)


def test_feature_vector_order_is_fixed():
    f = _features(n_tokens=7, n_linked_tables=3)
    vec = f.to_vector()
    assert len(vec) == DifficultyFeatures.dim()
    assert vec[DifficultyFeatures.ORDER.index("n_tokens")] == 7.0


def test_extract_features_from_a_real_question(linker, normalizer, join_graph):
    from aegis_sql.router.features import extract_features

    nq = normalizer.normalize("각 지점에서 모집 실적 1위인 설계사를 알려줘")
    linked = linker.link(nq)
    f = extract_features(nq, linked, join_graph, None)
    assert f.n_tokens > 0
    assert f.n_linked_tables >= 1
    assert len(f.to_vector()) == DifficultyFeatures.dim()


def test_heuristic_difficulty_orders_questions(linker, normalizer, join_graph):
    from aegis_sql.router.features import extract_features, heuristic_difficulty

    def score(q):
        nq = normalizer.normalize(q)
        return heuristic_difficulty(extract_features(nq, linker.link(nq), join_graph, None))

    easy = score("전체 계약 건수")
    hard = score("각 상품유형별로 전체 평균보다 계약이 많은 채널의 비중을 지점별로 비교해줘")
    assert 0.0 <= easy <= 1.0 and 0.0 <= hard <= 1.0
    assert hard > easy


def test_cascade_prefers_cheap_tier_for_easy_questions(settings):
    from aegis_sql.router.cascade import CascadeRouter

    router = CascadeRouter(settings, router=None, available_tiers={Tier.TEMPLATE, Tier.LLM})
    easy = _features(n_tokens=4, n_linked_tables=1, n_linked_columns=2, link_score_gap=0.5)
    decision = router.decide(easy)
    assert decision.tier in {Tier.TEMPLATE, Tier.SLM}
    assert decision.reason


def test_cascade_escalates_hard_questions(settings):
    from aegis_sql.router.cascade import CascadeRouter

    router = CascadeRouter(settings, router=None, available_tiers={Tier.TEMPLATE, Tier.LLM})
    hard = _features(
        n_tokens=32, n_linked_tables=5, n_linked_columns=30, max_join_depth=3,
        has_nested_cue=1, has_ratio_cue=1, has_aggregate_cue=1, ambiguity_score=0.3,
    )
    assert router.decide(hard).tier in {Tier.LLM, Tier.ENSEMBLE}


def test_cascade_respects_budget(settings):
    from aegis_sql.router.cascade import CascadeRouter

    router = CascadeRouter(settings, router=None, available_tiers={Tier.TEMPLATE, Tier.LLM})
    hard = _features(n_tokens=32, n_linked_tables=5, has_nested_cue=1, has_ratio_cue=1)
    starved = router.decide(hard, spent_usd=settings.router.budget_usd)
    assert starved.tier is not Tier.ENSEMBLE
    assert "예산" in starved.reason or "budget" in starved.reason.lower()


def test_cascade_only_picks_available_tiers(settings):
    from aegis_sql.router.cascade import CascadeRouter

    router = CascadeRouter(settings, router=None, available_tiers={Tier.TEMPLATE})
    hard = _features(n_tokens=40, n_linked_tables=6, has_nested_cue=1)
    assert router.decide(hard).tier is Tier.TEMPLATE


def test_escalation_moves_up_one_step(settings):
    from aegis_sql.router.cascade import CascadeRouter

    router = CascadeRouter(settings, router=None, available_tiers={Tier.TEMPLATE, Tier.SLM, Tier.LLM})
    first = router.decide(_features(n_tokens=5, n_linked_tables=1))
    second = router.escalate(first, "실행 실패")
    assert second.tier != first.tier
    assert second.escalated_from is first.tier


def test_calibration_reduces_ece():
    from aegis_sql.router.calibrator import TemperatureCalibrator, expected_calibration_error

    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 500)
    # Deliberately over-confident probabilities — the classic uncalibrated sigmoid.
    probs = np.clip(labels * 0.94 + (1 - labels) * 0.06 + rng.normal(0, 0.12, 500), 0.01, 0.99)
    probs = np.clip(probs ** 0.35, 0.01, 0.99)
    before = expected_calibration_error(probs, labels)
    cal = TemperatureCalibrator()
    cal.fit(probs, labels)
    after = expected_calibration_error(cal.transform(probs), labels)
    assert after <= before + 1e-6


@pytest.mark.slow
def test_keras_and_numpy_router_agree(tmp_path):
    pytest.importorskip("tensorflow")
    from aegis_sql.router.features import heuristic_difficulty
    from aegis_sql.router.tf_router import DifficultyRouter, NumpyRouter

    rng = np.random.default_rng(20260824)
    n, d = 800, DifficultyFeatures.dim()
    X = rng.random((n, d)).astype("float32") * np.array([30, 5, 6, 40, 4] + [1] * (d - 5), dtype="float32")
    y = (X[:, 4] + X[:, 9] + rng.normal(0, 0.2, n) > 2.0).astype("int32")

    router = DifficultyRouter(tmp_path / "router")
    metrics = router.train(X, y, epochs=8)
    assert "auc" in metrics
    router.export_numpy()

    numpy_router = NumpyRouter.load(tmp_path / "router")
    assert numpy_router is not None
    keras_preds = router.model.predict(X[:64], verbose=0).ravel()
    numpy_preds = np.array([numpy_router.predict_proba(x) for x in X[:64]])
    assert float(np.max(np.abs(keras_preds - numpy_preds))) < 1e-5
    assert 0.0 <= heuristic_difficulty(DifficultyFeatures()) <= 1.0
