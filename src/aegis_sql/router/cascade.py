"""The cascade: spend frontier-model money only where it changes the answer.

Calling a frontier model on every question is the expensive, lazy design.  On
this benchmark most Korean analytics questions are a filter plus a GROUP BY over
one or two tables — a shape the deterministic template grammar emits exactly and
for free.  Paying ~$0.009 and ~2 s of latency for those is pure waste, and it is
waste that scales linearly with traffic.

So the cascade predicts *before generating* whether the cheap tier will get this
particular question right, and buys the expensive tier only for the ones where
it will not.  Four tiers, cheapest first:

    TEMPLATE  → deterministic grammar, $0, ~8 ms
    SLM       → in-house 15M-parameter model, $0 marginal, ~200 ms
    LLM       → hosted frontier model, ~$0.009, ~2 s
    ENSEMBLE  → n samples on the LLM tier + execution voting, ~n × $0.009

Three properties make this defensible rather than merely cheap:

* **One decision, not a retry chain.**  FrugalGPT-style cascades run the cheap
  model, inspect the answer and retry upward; that doubles latency on exactly
  the hard queries that were already the slowest.  Here the tier is predicted up
  front from features that cost nothing (the linker and the NLU layer already
  ran), and :meth:`CascadeRouter.escalate` exists only as the *recovery* path
  after execution actually failed.
* **The budget is a hard constraint, not a report.**  ``decide`` is given what
  has already been spent on this query and will never choose a tier whose
  expected cost exceeds what is left — it downgrades, records why, and says so
  in ``reason``.  An ensemble is first narrowed sample-by-sample before being
  abandoned, because 3 samples within budget beat 0.
* **Every decision explains itself.**  ``RouteDecision.reason`` is Korean,
  one line, and shown verbatim in the trace and the web console.  A router
  nobody can audit gets switched off by the first analyst who disagrees with it.

Availability is intersected in as well: a tier that has no working generator
(no API key, no trained checkpoint) is never selected, so a fresh checkout with
zero credentials still answers — on TEMPLATE, with a reason saying so.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from aegis_sql.config import PROJECT_ROOT
from aegis_sql.llm.base import estimate_cost
from aegis_sql.observability.logging import get_logger
from aegis_sql.router.calibrator import CALIBRATOR_FILE, TemperatureCalibrator
from aegis_sql.router.features import heuristic_difficulty, top_drivers
from aegis_sql.types import DifficultyFeatures, RouteDecision, Tier

if TYPE_CHECKING:  # pragma: no cover - typing only
    from aegis_sql.config import Settings
    from aegis_sql.router.tf_router import NumpyRouter

log = get_logger("router.cascade")

#: Cheapest → most expensive.  Escalation walks this list rightwards.
LADDER: tuple[Tier, ...] = (Tier.TEMPLATE, Tier.SLM, Tier.LLM, Tier.ENSEMBLE)

#: The template tier is chosen below ``escalate_threshold × this``.  It is a
#: fraction rather than a constant so that tuning the one documented knob in
#: ``configs/default.yaml`` moves both boundaries coherently.
TEMPLATE_MAX_RATIO = 0.55

#: Token estimate for one LLM generation: an mschema schema card plus few-shots
#: is ~1.8k input tokens and a single SELECT is ~220 output tokens on this
#: benchmark.  Only ever used for the *pre-flight* budget check — the number
#: billed downstream is the provider's real usage, never this one.
EST_PROMPT_TOKENS = 1800
EST_COMPLETION_TOKENS = 220

#: Feature name → the word that goes in the Korean explanation.
_DRIVER_LABEL: dict[str, str] = {
    "has_nested_cue": "중첩질의",
    "has_ratio_cue": "비율계산",
    "has_set_op_cue": "집합연산",
    "has_ranking_cue": "순위",
    "has_comparison_cue": "범위조건",
    "has_temporal_cue": "기간",
    "has_aggregate_cue": "집계",
    "max_join_depth": "조인깊이",
    "n_linked_tables": "다중테이블",
    "n_linked_columns": "넓은컬럼범위",
    "schema_coverage": "스키마축소실패",
    "ambiguity_score": "질문모호성",
    "link_score_gap": "컬럼매핑경합",
    "link_score_mean": "링킹근거부족",
    "n_tokens": "긴질문",
    "n_value_candidates": "다수리터럴",
}


@dataclass(slots=True)
class RoutePolicy:
    """The knobs that govern one routing decision (a view over ``RouterConfig``)."""

    escalate_threshold: float = 0.55
    ensemble_threshold: float = 0.35
    budget_usd: float = 0.05
    allow_llm: bool = True
    allow_slm: bool = True

    @classmethod
    def from_settings(cls, settings: Settings) -> RoutePolicy:
        cfg = settings.router
        return cls(
            escalate_threshold=float(cfg.escalate_threshold),
            ensemble_threshold=float(cfg.ensemble_threshold),
            budget_usd=float(cfg.budget_usd),
        )

    @property
    def template_max(self) -> float:
        return self.escalate_threshold * TEMPLATE_MAX_RATIO

    def permits(self, tier: Tier) -> bool:
        if tier is Tier.SLM:
            return self.allow_slm
        if tier in (Tier.LLM, Tier.ENSEMBLE):
            return self.allow_llm
        return True


class CascadeRouter:
    """Chooses the tier for a question, and escalates when one has failed."""

    __slots__ = ("settings", "policy", "_router", "_calibrator", "_available")

    def __init__(
        self,
        settings: Settings,
        router: NumpyRouter | None = None,
        calibrator: TemperatureCalibrator | None = None,
        available_tiers: set[Tier] | None = None,
    ) -> None:
        self.settings = settings
        self.policy = RoutePolicy.from_settings(settings)
        self._router = router
        self._calibrator = calibrator if calibrator is not None else self._autoload_calibrator(settings)
        self._available: set[Tier] = set(available_tiers) if available_tiers is not None else set(LADDER)

    @staticmethod
    def _autoload_calibrator(settings: Settings) -> TemperatureCalibrator | None:
        """Pick up ``calibrator.json`` next to whichever weights actually loaded.

        This must follow the same fallback order as ``Engine._load_router``:
        freshly trained weights first, the checked-in ``models/router`` second.
        Looking only at the configured dir meant a fresh clone ran the learned
        router *uncalibrated* — the weights were found in ``models/router`` but
        the calibrator beside them was never read, so published confidences did
        not reproduce for anyone who cloned the repo.
        """
        for directory in (Path(settings.router.model_dir), PROJECT_ROOT / "models" / "router"):
            try:
                calibrator = TemperatureCalibrator.load(directory / CALIBRATOR_FILE)
            except Exception as exc:  # pragma: no cover - never block engine start-up
                log.debug("calibrator unavailable", path=str(directory), error=str(exc))
                continue
            if calibrator is not None:
                log.debug("calibrator loaded", path=str(directory))
                return calibrator
        return None

    # -- introspection ----------------------------------------------------- #

    @property
    def model(self) -> NumpyRouter | None:
        """The learned router, or ``None`` when running on the heuristic.

        Named ``model`` because ``/health`` reports it as ``router_loaded``.
        """
        return self._router

    @property
    def available_tiers(self) -> set[Tier]:
        return set(self._available)

    def _allowed(self) -> list[Tier]:
        return [t for t in LADDER if t in self._available and self.policy.permits(t)]

    # -- cost --------------------------------------------------------------- #

    def expected_cost(self, tier: Tier, n_samples: int = 1) -> float:
        """Pre-flight USD estimate for answering one question on ``tier``."""
        if tier in (Tier.TEMPLATE, Tier.SLM):
            return 0.0
        unit = estimate_cost(self.settings.generation.model, EST_PROMPT_TOKENS, EST_COMPLETION_TOKENS)
        return unit * max(1, n_samples) if tier is Tier.ENSEMBLE else unit

    # -- the decision ------------------------------------------------------- #

    def decide(self, features: DifficultyFeatures, spent_usd: float = 0.0) -> RouteDecision:
        """Pick a tier for one question, given what this query has already cost.

        ``RouteDecision.difficulty`` is the calibrated ``P(the cheap tier fails)``
        and ``confidence`` is its complement, so both numbers shown in the trace
        are on the same scale as the thresholds that produced the decision.
        """
        # The number every threshold below is compared against is the *calibrated*
        # one — an uncalibrated sigmoid makes `escalate_threshold` mean whatever
        # the last training run happened to make it mean (see calibrator.py).
        difficulty = self._calibrate(self._difficulty(features))
        confidence = 1.0 - difficulty

        allowed = self._allowed()
        desired, intent = self._desired_tier(features, difficulty, confidence, allowed)
        n_samples = self._ensemble_samples() if desired is Tier.ENSEMBLE else 1

        remaining = max(0.0, self.policy.budget_usd - spent_usd)
        tier, n_samples, notes = self._fit(desired, n_samples, allowed, remaining)

        decision = RouteDecision(
            tier=tier,
            difficulty=round(difficulty, 4),
            confidence=round(confidence, 4),
            reason=self._reason(features, difficulty, confidence, tier, n_samples, intent, notes),
            n_samples=n_samples,
            features=features,
        )
        log.debug(
            "routed",
            tier=tier.value,
            difficulty=round(difficulty, 3),
            confidence=round(confidence, 3),
            samples=n_samples,
            learned=self._router is not None,
        )
        return decision

    def escalate(self, current: RouteDecision, reason: str) -> RouteDecision:
        """Move one rung up the ladder after a failure downstream.

        Used when generation produced nothing executable, the guard rewrote the
        statement into something empty, or ensemble agreement was too low to
        trust.  Returns a decision on the *same* tier (with an explanatory
        reason) when there is nothing left above — the pipeline treats that as
        "stop retrying".
        """
        allowed = self._allowed()
        start = LADDER.index(current.tier)
        target = next((t for t in LADDER[start + 1 :] if t in allowed), None)
        if target is None:
            return RouteDecision(
                tier=current.tier,
                difficulty=current.difficulty,
                confidence=current.confidence,
                reason=f"{reason} · 상위 티어 없음 → {current.tier.value} 유지",
                escalated_from=current.escalated_from,
                n_samples=current.n_samples,
                features=current.features,
            )

        n_samples = self._ensemble_samples() if target is Tier.ENSEMBLE else 1
        cost = self.expected_cost(target, n_samples)
        if cost > self.policy.budget_usd:
            return RouteDecision(
                tier=current.tier,
                difficulty=current.difficulty,
                confidence=current.confidence,
                reason=f"{reason} · 예산 한도(${self.policy.budget_usd:.3f}) 초과로 상향 불가",
                escalated_from=current.escalated_from,
                n_samples=current.n_samples,
                features=current.features,
            )

        return RouteDecision(
            tier=target,
            difficulty=current.difficulty,
            confidence=current.confidence,
            reason=f"{reason} → {target.value} 로 상향" + (f" ({n_samples}샘플)" if n_samples > 1 else ""),
            escalated_from=current.tier,
            n_samples=n_samples,
            features=current.features,
        )

    # -- internals ---------------------------------------------------------- #

    def _difficulty(self, features: DifficultyFeatures) -> float:
        if self._router is None:
            return heuristic_difficulty(features)
        try:
            return float(min(1.0, max(0.0, self._router.predict_proba(features))))
        except Exception as exc:  # pragma: no cover - a bad artefact must not 500
            log.warning("learned router failed, using heuristic", error=str(exc))
            return heuristic_difficulty(features)

    def _calibrate(self, difficulty: float) -> float:
        """Apply the fitted calibration map, but only to the model it was fitted on.

        A temperature and bias fitted on the network's logits say nothing about
        the heuristic's weighted sum, so the heuristic path is left alone rather
        than warped by a map that was never estimated for it.
        """
        if self._calibrator is None or self._router is None:
            return difficulty
        return float(min(1.0, max(0.0, self._calibrator.transform_one(difficulty))))

    def _ensemble_samples(self) -> int:
        return max(2, int(self.settings.generation.ensemble_samples))

    def _desired_tier(
        self,
        features: DifficultyFeatures,
        difficulty: float,
        confidence: float,
        allowed: list[Tier],
    ) -> tuple[Tier, str]:
        """The tier the model *wants*, before availability and budget are applied."""
        if (
            difficulty <= self.policy.template_max
            and Tier.TEMPLATE in allowed
            and not features.has_nested_cue
        ):
            return Tier.TEMPLATE, "결정적 템플릿으로 충분"
        if difficulty < self.policy.escalate_threshold:
            return Tier.SLM, "자체 sLLM 범위"
        if confidence < self.policy.ensemble_threshold:
            return Tier.ENSEMBLE, "신뢰도 낮아 자기일관성 투표"
        return Tier.LLM, "난이도 높아 프런티어 모델"

    def _fit(
        self,
        desired: Tier,
        n_samples: int,
        allowed: list[Tier],
        remaining: float,
    ) -> tuple[Tier, int, list[str]]:
        """Apply availability, then the budget.  Returns the final tier and why."""
        notes: list[str] = []
        if not allowed:  # pragma: no cover - an engine with no generators at all
            return desired, n_samples, ["사용 가능한 생성기 없음"]

        tier = _nearest_available(desired, allowed)
        if tier is not desired:
            notes.append(f"{desired.value} 티어 미가용 → {tier.value}")
            n_samples = self._ensemble_samples() if tier is Tier.ENSEMBLE else 1

        # An ensemble that does not fit is narrowed before it is abandoned:
        # three samples inside the budget are worth more than none.
        if tier is Tier.ENSEMBLE and self.expected_cost(tier, n_samples) > remaining:
            affordable = [n for n in range(n_samples, 1, -1) if self.expected_cost(tier, n) <= remaining]
            if affordable:
                notes.append(f"잔여 예산 ${remaining:.4f} → 앙상블 {affordable[0]}샘플로 축소")
                return tier, affordable[0], notes

        index = LADDER.index(tier)
        while self.expected_cost(tier, n_samples) > remaining:
            cheaper = next((t for t in reversed(LADDER[:index]) if t in allowed), None)
            if cheaper is None:
                notes.append(f"잔여 예산 ${remaining:.4f} 미만이지만 더 저렴한 티어가 없어 {tier.value} 유지")
                break
            notes.append(f"잔여 예산 ${remaining:.4f} → {tier.value} 대신 {cheaper.value}")
            tier, index = cheaper, LADDER.index(cheaper)
            n_samples = self._ensemble_samples() if tier is Tier.ENSEMBLE else 1
        return tier, n_samples, notes

    def _reason(
        self,
        features: DifficultyFeatures,
        difficulty: float,
        confidence: float,
        tier: Tier,
        n_samples: int,
        intent: str,
        notes: list[str],
    ) -> str:
        if self._router is None:
            source = "휴리스틱"
        else:
            source = "학습 라우터+보정" if self._calibrator is not None else "학습 라우터(미보정)"
        drivers = [_DRIVER_LABEL.get(name, name) for name in top_drivers(features, k=2)]
        driver_text = f"({'·'.join(drivers)})" if drivers else ""
        samples = f" {n_samples}샘플" if n_samples > 1 else ""
        head = (
            f"난이도 {difficulty:.2f}{driver_text} · 신뢰도 {confidence:.2f} [{source}]"
            f" · {intent} → {tier.value}{samples}"
        )
        return " · ".join([head, *notes])


def _nearest_available(desired: Tier, allowed: list[Tier]) -> Tier:
    """Best allowed tier at or below ``desired``; failing that, the cheapest above.

    Downgrading is preferred because a cheaper tier is always affordable and
    always fast — and the post-hoc :meth:`CascadeRouter.escalate` path exists to
    recover if it turns out to have been too weak.
    """
    index = LADDER.index(desired)
    below = [t for t in LADDER[: index + 1] if t in allowed]
    if below:
        return below[-1]
    above = [t for t in LADDER[index + 1 :] if t in allowed]
    return above[0] if above else desired
