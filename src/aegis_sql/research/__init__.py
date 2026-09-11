"""Executable research adapters derived from paper-review hypotheses.

These modules deliberately stay outside the default production path until a
paper-driven hypothesis has enough external validation to justify changing the
core engine.  The goal is to make research claims executable without silently
turning a 30-item probe into a general production guarantee.
"""

from aegis_sql.research.capability_engine import CapabilityAwareEngine, CapabilityAwareResult
from aegis_sql.research.selective_resampling import (
    ResamplingDecision,
    ResamplingSignal,
    ResamplingThresholds,
    decide_selective_resampling,
    evaluate_policy,
    ids_sha256,
    summarize_live_experiment,
    validate_live_evidence,
)

__all__ = [
    "CapabilityAwareEngine",
    "CapabilityAwareResult",
    "ResamplingDecision",
    "ResamplingSignal",
    "ResamplingThresholds",
    "decide_selective_resampling",
    "evaluate_policy",
    "ids_sha256",
    "summarize_live_experiment",
    "validate_live_evidence",
]
