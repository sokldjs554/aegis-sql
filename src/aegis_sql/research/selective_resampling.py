"""R³-SQL-inspired selective resampling policy.

R³-SQL uses a learned/agentic judgment over a candidate pool.  AEGIS does not
claim to reproduce that mechanism here.  This module implements a conservative,
fully observable approximation using signals the current engine already records:
router confidence, execution-result group count and winner agreement.

The policy is deliberately side-effect free.  It can be evaluated offline on
recorded candidate pools before anyone wires it into the production router.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class ResamplingThresholds:
    confidence: float = 0.60
    agreement: float = 0.60
    min_groups: int = 2
    min_candidates: int = 2


@dataclass(frozen=True, slots=True)
class ResamplingSignal:
    route_confidence: float
    groups: int
    agreement: float
    candidate_count: int


@dataclass(frozen=True, slots=True)
class ResamplingDecision:
    trigger: bool
    reason: str
    uncertainty: float
    dispersion: float


def decide_selective_resampling(
    signal: ResamplingSignal,
    thresholds: ResamplingThresholds = ResamplingThresholds(),
) -> ResamplingDecision:
    """Decide whether a candidate pool is uncertain enough to resample.

    Both uncertainty and disagreement are required.  This avoids paying for
    another generation merely because two equivalent candidates happened to be
    sampled, or because a router confidence is low while every execution result
    agrees.
    """

    if signal.candidate_count < thresholds.min_candidates:
        return ResamplingDecision(False, "candidate pool too small", 0.0, 0.0)

    uncertainty = max(0.0, thresholds.confidence - signal.route_confidence)
    dispersion = max(0.0, thresholds.agreement - signal.agreement)
    enough_groups = signal.groups >= thresholds.min_groups
    trigger = uncertainty > 0.0 and dispersion > 0.0 and enough_groups

    if trigger:
        reason = (
            f"low router confidence ({signal.route_confidence:.3f}) + "
            f"dispersed execution groups ({signal.groups}, agreement={signal.agreement:.3f})"
        )
    elif not enough_groups:
        reason = "execution results do not form multiple groups"
    elif uncertainty <= 0.0:
        reason = "router confidence is above the resampling threshold"
    else:
        reason = "candidate agreement is above the resampling threshold"

    return ResamplingDecision(
        trigger=trigger,
        reason=reason,
        uncertainty=round(uncertainty, 4),
        dispersion=round(dispersion, 4),
    )


def evaluate_policy(
    rows: Iterable[dict[str, Any]],
    thresholds: ResamplingThresholds = ResamplingThresholds(),
) -> dict[str, Any]:
    """Evaluate trigger rate and, when present, counterfactual outcome fields.

    Each row must contain ``route_confidence``, ``groups``, ``agreement`` and
    ``candidate_count``.  Optional fields ``baseline_correct`` and
    ``resampled_correct`` enable a counterfactual accuracy calculation where the
    resampled answer is used only on triggered rows.  Optional cost/latency
    fields are accumulated in the same way.
    """

    total = triggered = 0
    baseline_correct = policy_correct = comparable = 0
    baseline_cost = policy_cost = 0.0
    baseline_latency = policy_latency = 0.0
    decisions: list[dict[str, Any]] = []

    for row in rows:
        total += 1
        signal = ResamplingSignal(
            route_confidence=float(row["route_confidence"]),
            groups=int(row["groups"]),
            agreement=float(row["agreement"]),
            candidate_count=int(row["candidate_count"]),
        )
        decision = decide_selective_resampling(signal, thresholds)
        triggered += int(decision.trigger)

        item: dict[str, Any] = {
            "id": row.get("id", total),
            "trigger": decision.trigger,
            "reason": decision.reason,
            "uncertainty": decision.uncertainty,
            "dispersion": decision.dispersion,
        }

        if "baseline_correct" in row and "resampled_correct" in row:
            comparable += 1
            base_ok = bool(row["baseline_correct"])
            resampled_ok = bool(row["resampled_correct"])
            chosen_ok = resampled_ok if decision.trigger else base_ok
            baseline_correct += int(base_ok)
            policy_correct += int(chosen_ok)
            item["baseline_correct"] = base_ok
            item["policy_correct"] = chosen_ok

        base_cost = float(row.get("baseline_cost_usd", 0.0))
        extra_cost = float(row.get("resample_extra_cost_usd", 0.0)) if decision.trigger else 0.0
        baseline_cost += base_cost
        policy_cost += base_cost + extra_cost

        base_latency = float(row.get("baseline_latency_ms", 0.0))
        extra_latency = float(row.get("resample_extra_latency_ms", 0.0)) if decision.trigger else 0.0
        baseline_latency += base_latency
        policy_latency += base_latency + extra_latency
        decisions.append(item)

    result: dict[str, Any] = {
        "items": total,
        "triggered": triggered,
        "trigger_rate": triggered / total if total else 0.0,
        "thresholds": {
            "confidence": thresholds.confidence,
            "agreement": thresholds.agreement,
            "min_groups": thresholds.min_groups,
            "min_candidates": thresholds.min_candidates,
        },
        "baseline_cost_usd": round(baseline_cost, 6),
        "policy_cost_usd": round(policy_cost, 6),
        "baseline_latency_ms_sum": round(baseline_latency, 2),
        "policy_latency_ms_sum": round(policy_latency, 2),
        "decisions": decisions,
    }
    if comparable:
        result["comparable_items"] = comparable
        result["baseline_accuracy"] = baseline_correct / comparable
        result["policy_accuracy"] = policy_correct / comparable
        result["accuracy_delta"] = (policy_correct - baseline_correct) / comparable
    return result
