"""Runtime guardrails for blocking query execution.

The query engine is synchronous and may call remote LLMs, so FastAPI runs it in
an executor thread.  Timing out the HTTP await does *not* stop that thread.
This gate therefore keeps a capacity slot until the real worker finishes, even
when the caller has already received a timeout or disconnected.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TypeVar

from aegis_sql.observability.metrics import QUERY_RUNTIME_REJECTIONS

_T = TypeVar("_T")


class QueryCapacityExceeded(RuntimeError):
    """Raised when all query worker slots are occupied."""


class QueryTimedOut(RuntimeError):
    """Raised when a query exceeds the caller-facing runtime budget."""


class QueryRuntimeGate:
    """Fail fast at capacity and bound how long callers await a query worker."""

    def __init__(self, *, max_concurrent: int, timeout_s: float) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.timeout_s = timeout_s
        self._slots = asyncio.BoundedSemaphore(max_concurrent)

    async def run(self, call: Callable[[], _T]) -> _T:
        """Run one blocking call without oversubscribing the worker capacity.

        A timed-out/cancelled executor Future cannot be force-killed safely.
        ``shield`` keeps it alive, and the slot is released by its completion
        callback rather than when the HTTP request stops waiting.
        """
        if self._slots.locked():
            QUERY_RUNTIME_REJECTIONS.labels(reason="capacity").inc()
            raise QueryCapacityExceeded("all query worker slots are occupied")

        await self._slots.acquire()
        loop = asyncio.get_running_loop()
        worker = loop.run_in_executor(None, call)
        release_deferred = False

        try:
            return await asyncio.wait_for(asyncio.shield(worker), timeout=self.timeout_s)
        except asyncio.TimeoutError as exc:
            release_deferred = True
            worker.add_done_callback(lambda _future: self._slots.release())
            QUERY_RUNTIME_REJECTIONS.labels(reason="timeout").inc()
            raise QueryTimedOut(f"query exceeded {self.timeout_s:g}s") from exc
        except asyncio.CancelledError:
            release_deferred = True
            worker.add_done_callback(lambda _future: self._slots.release())
            raise
        finally:
            if not release_deferred:
                self._slots.release()
