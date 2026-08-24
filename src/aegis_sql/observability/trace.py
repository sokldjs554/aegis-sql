"""A tiny hierarchical tracer.

Every stage of the pipeline opens a span; the resulting tree is returned inside
:class:`~aegis_sql.types.AnswerBundle` and rendered by both the CLI and the web
console.  It is intentionally dependency-free so that traces work in CI, in the
evaluation harness and inside training loops without an OTel collector.
"""

from __future__ import annotations

import contextvars
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from aegis_sql.types import Span, now_ms

_current_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar("aegis_span", default=None)
_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("aegis_trace_id", default="")


class Tracer:
    """Collects a span tree for one query."""

    def __init__(self, name: str = "query", trace_id: str | None = None) -> None:
        self.trace_id = trace_id or uuid.uuid4().hex[:12]
        self.root = Span(name=name, start_ms=now_ms())
        self._stack: list[Span] = [self.root]

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        span = Span(name=name, start_ms=now_ms(), attributes=dict(attributes))
        self._stack[-1].children.append(span)
        self._stack.append(span)
        token = _current_span.set(span)
        try:
            yield span
        except Exception as exc:  # pragma: no cover - re-raised
            span.attributes["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.end_ms = now_ms()
            self._stack.pop()
            _current_span.reset(token)

    def event(self, name: str, **attributes: Any) -> None:
        span = Span(name=name, start_ms=now_ms(), attributes=dict(attributes))
        span.end_ms = span.start_ms
        self._stack[-1].children.append(span)

    def annotate(self, **attributes: Any) -> None:
        self._stack[-1].attributes.update(attributes)

    def finish(self) -> Span:
        self.root.end_ms = now_ms()
        return self.root

    @property
    def elapsed_ms(self) -> float:
        return now_ms() - self.root.start_ms


@contextmanager
def trace_context(trace_id: str) -> Iterator[str]:
    token = _trace_id.set(trace_id)
    try:
        yield trace_id
    finally:
        _trace_id.reset(token)


def current_trace_id() -> str:
    return _trace_id.get()


def render_span(span: Span, indent: int = 0, out: list[str] | None = None) -> str:
    """Pretty ASCII rendering of a span tree (used by the CLI)."""
    out = [] if out is None else out
    pad = "  " * indent
    attrs = " ".join(f"{k}={v}" for k, v in span.attributes.items() if not isinstance(v, (dict, list)))
    out.append(f"{pad}├─ {span.name} ({span.duration_ms:.1f}ms) {attrs}".rstrip())
    for child in span.children:
        render_span(child, indent + 1, out)
    return "\n".join(out)
