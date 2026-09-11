"""R³-SQL-inspired selective resampling policy and evidence summary.

R³-SQL uses a learned/agentic judgment over a candidate pool. AEGIS does not
claim to reproduce that mechanism here. This module implements a conservative,
fully observable approximation using signals the current engine already records:
router confidence, execution-result group count and winner agreement.

The trigger remains side-effect free. Live candidate generation is owned by
``scripts/run_selective_resampling.py``; this module only decides and summarizes
what was actually observed. Keeping those responsibilities separate prevents a
synthetic/offline row from being mistaken for a paid full-benchmark experiment.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any


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

    Both uncertainty and disagreement are required. This avoids paying for
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
    """Summarize paired baseline and *observed* selective-resampling rows.

    Required signal fields are ``route_confidence``, ``groups``, ``agreement``
    and ``candidate_count``. ``baseline_correct`` is sufficient on a
    non-triggered row; a triggered row is comparable only when
    ``resampled_correct`` is a real boolean. Missing resampled outcomes are
    never coerced to failure, which is important for interrupted paid runs.
    """

    materialized = list(rows)
    total = triggered = eligible = 0
    baseline_correct = policy_correct = comparable = 0
    gains = regressions = unchanged = 0
    baseline_cost = extra_cost = 0.0
    baseline_latencies: list[float] = []
    extra_latencies: list[float] = []
    policy_latencies: list[float] = []
    baseline_completed_samples = extra_completed_samples = 0
    baseline_requested_samples = extra_requested_samples = 0
    decisions: list[dict[str, Any]] = []
    difficulty_rows: dict[str, list[tuple[bool, bool]]] = defaultdict(list)

    for index, row in enumerate(materialized, 1):
        total += 1
        signal = _signal(row)
        decision = decide_selective_resampling(signal, thresholds)
        triggered += int(decision.trigger)
        eligible += int(signal.candidate_count >= thresholds.min_candidates)

        item: dict[str, Any] = {
            "id": row.get("id", index),
            "trigger": decision.trigger,
            "reason": decision.reason,
            "uncertainty": decision.uncertainty,
            "dispersion": decision.dispersion,
        }

        base_observed = isinstance(row.get("baseline_correct"), bool)
        resampled_observed = isinstance(row.get("resampled_correct"), bool)
        if base_observed and (not decision.trigger or resampled_observed):
            comparable += 1
            base_ok = bool(row["baseline_correct"])
            chosen_ok = bool(row["resampled_correct"]) if decision.trigger else base_ok
            baseline_correct += int(base_ok)
            policy_correct += int(chosen_ok)
            gains += int(not base_ok and chosen_ok)
            regressions += int(base_ok and not chosen_ok)
            unchanged += int(base_ok == chosen_ok)
            item["baseline_correct"] = base_ok
            item["policy_correct"] = chosen_ok
            difficulty_rows[str(row.get("difficulty", "unknown"))].append((base_ok, chosen_ok))

        base_cost = _nonnegative_float(row.get("baseline_cost_usd", 0.0))
        added_cost = (
            _nonnegative_float(row.get("resample_extra_cost_usd", 0.0))
            if decision.trigger
            else 0.0
        )
        baseline_cost += base_cost
        extra_cost += added_cost

        base_latency = _nonnegative_float(row.get("baseline_latency_ms", 0.0))
        added_latency = (
            _nonnegative_float(row.get("resample_extra_latency_ms", 0.0))
            if decision.trigger
            else 0.0
        )
        baseline_latencies.append(base_latency)
        policy_latencies.append(base_latency + added_latency)
        if decision.trigger:
            extra_latencies.append(added_latency)

        baseline_generation = _generation(row.get("baseline"))
        resample_generation = _generation(row.get("resample"))
        baseline_requested_samples += _int_or_zero(baseline_generation.get("requested_samples", 0))
        baseline_completed_samples += _int_or_zero(baseline_generation.get("completed_samples", 0))
        if decision.trigger:
            extra_requested_samples += _int_or_zero(resample_generation.get("requested_samples", 0))
            extra_completed_samples += _int_or_zero(resample_generation.get("completed_samples", 0))
        decisions.append(item)

    policy_cost = baseline_cost + extra_cost
    result: dict[str, Any] = {
        "items": total,
        "eligible_items": eligible,
        "triggered": triggered,
        "trigger_rate": _ratio(triggered, total),
        "eligible_trigger_rate": _ratio(triggered, eligible),
        "thresholds": _threshold_dict(thresholds),
        "cost_usd": {
            "baseline_total": round(baseline_cost, 6),
            "extra_total": round(extra_cost, 6),
            "policy_total": round(policy_cost, 6),
            "extra_per_trigger": round(extra_cost / triggered, 6) if triggered else 0.0,
            "extra_per_query": round(extra_cost / total, 6) if total else 0.0,
            "increase_ratio": _ratio(extra_cost, baseline_cost),
        },
        "latency_ms": {
            "baseline_sum": round(sum(baseline_latencies), 2),
            "baseline_p50": _percentile(baseline_latencies, 0.50),
            "baseline_p95": _percentile(baseline_latencies, 0.95),
            "extra_sum": round(sum(extra_latencies), 2),
            "extra_triggered_p50": _percentile(extra_latencies, 0.50),
            "extra_triggered_p95": _percentile(extra_latencies, 0.95),
            "policy_sum": round(sum(policy_latencies), 2),
            "policy_p50": _percentile(policy_latencies, 0.50),
            "policy_p95": _percentile(policy_latencies, 0.95),
        },
        "samples": {
            "baseline_requested": baseline_requested_samples,
            "baseline_completed": baseline_completed_samples,
            "extra_requested": extra_requested_samples,
            "extra_completed": extra_completed_samples,
        },
        # Backward-compatible aggregate keys used by the original offline CLI.
        "baseline_cost_usd": round(baseline_cost, 6),
        "policy_cost_usd": round(policy_cost, 6),
        "baseline_latency_ms_sum": round(sum(baseline_latencies), 2),
        "policy_latency_ms_sum": round(sum(policy_latencies), 2),
        "decisions": decisions,
    }
    if comparable:
        baseline_accuracy = baseline_correct / comparable
        policy_accuracy = policy_correct / comparable
        result.update(
            {
                "comparable_items": comparable,
                "baseline_correct": baseline_correct,
                "policy_correct": policy_correct,
                "baseline_accuracy": baseline_accuracy,
                "policy_accuracy": policy_accuracy,
                "accuracy_delta": policy_accuracy - baseline_accuracy,
                "accuracy_delta_pp": (policy_accuracy - baseline_accuracy) * 100.0,
                "transitions": {
                    "gains": gains,
                    "regressions": regressions,
                    "unchanged": unchanged,
                },
                "difficulty": {
                    name: _difficulty_summary(values)
                    for name, values in sorted(difficulty_rows.items())
                },
            }
        )
    return result


def summarize_live_experiment(
    rows: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
    thresholds: ResamplingThresholds = ResamplingThresholds(),
    *,
    expected_items: int = 90,
) -> dict[str, Any]:
    """Build a strict evidence report without hiding incomplete live runs.

    ``portfolio_evidence_ready`` becomes true only for a clean, full, hosted
    provider run whose row count, hashes, provider identity and observed
    resampling outcomes all pass validation. The metrics are still returned
    when validation fails so smoke/partial runs remain useful for debugging.
    """

    metrics = evaluate_policy(rows, thresholds)
    errors = validate_live_evidence(rows, manifest, thresholds, expected_items=expected_items)
    return {
        "schema_version": 1,
        "experiment": "r3_selective_resampling",
        "portfolio_evidence_ready": not errors,
        "evidence_errors": errors,
        "manifest": manifest,
        "metrics": metrics,
    }


def validate_live_evidence(
    rows: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
    thresholds: ResamplingThresholds = ResamplingThresholds(),
    *,
    expected_items: int = 90,
) -> list[str]:
    """Return every reason a run is not publishable evidence."""

    errors: list[str] = []
    provider_requested = str(manifest.get("provider_requested", ""))
    provider_actual = str(manifest.get("provider_actual", ""))
    model_actual = str(manifest.get("model_actual", ""))
    if manifest.get("run_kind") != "full":
        errors.append("run_kind must be 'full'")
    if provider_requested not in {"anthropic", "openai"}:
        errors.append("an explicit hosted provider is required")
    if provider_actual != provider_requested:
        errors.append("requested and actual providers differ")
    if not model_actual or model_actual == "mock":
        errors.append("an actual hosted model id is required")

    git = _mapping(manifest.get("git"))
    if not git.get("commit"):
        errors.append("git commit is missing")
    if git.get("dirty") is not False:
        errors.append("git worktree must be clean")

    benchmark = _mapping(manifest.get("benchmark"))
    database = _mapping(manifest.get("database"))
    if benchmark.get("items") != expected_items:
        errors.append(f"manifest benchmark item count must be {expected_items}")
    if not _is_sha256(benchmark.get("sha256")):
        errors.append("benchmark SHA-256 is missing or malformed")
    if not _is_sha256(database.get("sha256")):
        errors.append("database SHA-256 is missing or malformed")
    if not database.get("schema_fingerprint"):
        errors.append("schema fingerprint is missing")
    if not manifest.get("prompt_manifest"):
        errors.append("prompt manifest is missing")
    if not manifest.get("router_artifacts_sha256"):
        errors.append("router artifact hashes are missing")
    runtime = _mapping(manifest.get("runtime"))
    if runtime.get("pythonhashseed") != "0":
        errors.append("PYTHONHASHSEED must be 0")
    if manifest.get("thresholds") != _threshold_dict(thresholds):
        errors.append("summary thresholds differ from the frozen manifest")
    if not 0.0 <= thresholds.confidence <= 1.0:
        errors.append("confidence threshold must be in [0, 1]")
    if not 0.0 <= thresholds.agreement <= 1.0:
        errors.append("agreement threshold must be in [0, 1]")
    if thresholds.min_groups < 1 or thresholds.min_candidates < 1:
        errors.append("minimum group/candidate thresholds must be positive")

    protocol = _mapping(manifest.get("protocol"))
    expected_resample_samples = _int_or_zero(protocol.get("resample_samples", 0))
    if expected_resample_samples < 2:
        errors.append("manifest resample sample count must be at least two")

    if len(rows) != expected_items:
        errors.append(f"raw row count must be {expected_items}, got {len(rows)}")
    ids = [str(row.get("id", "")) for row in rows]
    if not all(ids) or len(set(ids)) != len(ids):
        errors.append("row ids must be present and unique")
    selected_hash = benchmark.get("selected_ids_sha256")
    if selected_hash != ids_sha256(ids):
        errors.append("row ids do not match the frozen selected-id hash")

    observed_resamples = 0
    for index, row in enumerate(rows, 1):
        label = str(row.get("id") or index)
        try:
            signal = _signal(row)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{label}: invalid trigger signal ({exc})")
            continue
        if not 0.0 <= signal.route_confidence <= 1.0:
            errors.append(f"{label}: route_confidence must be in [0, 1]")
        if not 0.0 <= signal.agreement <= 1.0:
            errors.append(f"{label}: agreement must be in [0, 1]")
        if signal.groups < 0 or signal.candidate_count < 0:
            errors.append(f"{label}: groups and candidate_count must be nonnegative")

        decision = decide_selective_resampling(signal, thresholds)
        if row.get("trigger") is not decision.trigger:
            errors.append(f"{label}: stored trigger disagrees with the frozen policy")
        if not isinstance(row.get("baseline_correct"), bool):
            errors.append(f"{label}: baseline_correct is not observed")
        for key in (
            "baseline_cost_usd",
            "resample_extra_cost_usd",
            "baseline_latency_ms",
            "resample_extra_latency_ms",
        ):
            if not _is_nonnegative_number(row.get(key)):
                errors.append(f"{label}: {key} must be a finite nonnegative number")

        baseline = row.get("baseline")
        if not isinstance(baseline, dict):
            errors.append(f"{label}: baseline audit record is missing")
            continue
        generation = _generation(baseline)
        tier = str(baseline.get("tier", ""))
        if signal.candidate_count > 1 and not baseline.get("vote_observed"):
            errors.append(f"{label}: multi-candidate baseline has no observed vote stats")
        if tier in {"llm", "ensemble"}:
            requested = _int_or_zero(generation.get("requested_samples", 0))
            completed = _int_or_zero(generation.get("completed_samples", 0))
            if completed < 1:
                errors.append(f"{label}: hosted baseline generation returned no provider response")
            if completed != requested:
                errors.append(f"{label}: baseline provider response count is incomplete")

        if decision.trigger:
            observed_resamples += 1
            resample = row.get("resample")
            if not isinstance(resample, dict):
                errors.append(f"{label}: triggered row has no resampling audit record")
                continue
            resample_generation = _generation(resample)
            if resample.get("tier") != "ensemble":
                errors.append(f"{label}: resampling did not stay on the forced ensemble tier")
            requested = _int_or_zero(resample_generation.get("requested_samples", 0))
            completed = _int_or_zero(resample_generation.get("completed_samples", 0))
            if requested != expected_resample_samples:
                errors.append(f"{label}: resampling request count differs from the manifest")
            if completed != requested:
                errors.append(f"{label}: resampling provider response count is incomplete")
            if not isinstance(row.get("resampled_correct"), bool):
                errors.append(f"{label}: triggered resampled outcome is not observed")
        else:
            if row.get("resample") is not None:
                errors.append(f"{label}: non-triggered row unexpectedly contains a resample")
            if row.get("resampled_correct") is not None:
                errors.append(f"{label}: non-triggered row has a counterfactual resampled outcome")
            if row.get("resample_extra_cost_usd") != 0.0:
                errors.append(f"{label}: non-triggered row has extra cost")
            if row.get("resample_extra_latency_ms") != 0.0:
                errors.append(f"{label}: non-triggered row has extra latency")

    if rows and observed_resamples == 0:
        errors.append("the policy triggered no actual resampling calls")
    return errors


def ids_sha256(ids: Sequence[str]) -> str:
    """Stable digest of the ordered benchmark ids used by the run."""

    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def _signal(row: dict[str, Any]) -> ResamplingSignal:
    return ResamplingSignal(
        route_confidence=float(row["route_confidence"]),
        groups=int(row["groups"]),
        agreement=float(row["agreement"]),
        candidate_count=int(row["candidate_count"]),
    )


def _threshold_dict(thresholds: ResamplingThresholds) -> dict[str, Any]:
    return {
        "confidence": thresholds.confidence,
        "agreement": thresholds.agreement,
        "min_groups": thresholds.min_groups,
        "min_candidates": thresholds.min_candidates,
    }


def _generation(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    generation = value.get("generation")
    return generation if isinstance(generation, dict) else {}


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _nonnegative_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) and number >= 0.0 else 0.0


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return 0


def _is_nonnegative_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value)) and float(value) >= 0.0


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(char in "0123456789abcdef" for char in value.lower())


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 2)
    weight = position - lower
    return round(ordered[lower] * (1.0 - weight) + ordered[upper] * weight, 2)


def _difficulty_summary(values: Sequence[tuple[bool, bool]]) -> dict[str, Any]:
    total = len(values)
    baseline = sum(base for base, _ in values)
    policy = sum(chosen for _, chosen in values)
    return {
        "items": total,
        "baseline_correct": baseline,
        "policy_correct": policy,
        "baseline_accuracy": _ratio(baseline, total),
        "policy_accuracy": _ratio(policy, total),
        "delta_pp": _ratio(policy - baseline, total) * 100.0,
    }
