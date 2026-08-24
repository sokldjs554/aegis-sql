"""Prometheus metrics with a no-op fallback when prometheus_client is absent."""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - import guard
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

    _HAS_PROM = True
except Exception:  # pragma: no cover
    _HAS_PROM = False
    CONTENT_TYPE_LATEST = "text/plain"

    class _Noop:
        def __init__(self, *a: Any, **k: Any) -> None: ...
        def labels(self, *a: Any, **k: Any) -> _Noop:
            return self
        def inc(self, *a: Any, **k: Any) -> None: ...
        def observe(self, *a: Any, **k: Any) -> None: ...

    Counter = Histogram = _Noop  # type: ignore[assignment,misc]

    def generate_latest(*a: Any, **k: Any) -> bytes:  # type: ignore[misc]
        return b""


QUERIES = Counter("aegis_queries_total", "Queries processed", ["status", "tier"])
REPAIRS = Counter("aegis_repairs_total", "Self-repair attempts", ["fixed"])
GUARD_BLOCKS = Counter("aegis_guard_blocks_total", "Governance blocks", ["code"])
ESCALATIONS = Counter("aegis_escalations_total", "Cascade escalations", ["from_tier", "to_tier"])
LATENCY = Histogram(
    "aegis_query_latency_ms",
    "End-to-end query latency (ms)",
    ["tier"],
    buckets=(10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000),
)
STAGE_LATENCY = Histogram(
    "aegis_stage_latency_ms",
    "Per-stage latency (ms)",
    ["stage"],
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000),
)
TOKENS = Counter("aegis_tokens_total", "LLM tokens consumed", ["kind"])


def metrics_payload() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST


__all__ = [
    "QUERIES", "REPAIRS", "GUARD_BLOCKS", "ESCALATIONS", "LATENCY",
    "STAGE_LATENCY", "TOKENS", "metrics_payload",
]
