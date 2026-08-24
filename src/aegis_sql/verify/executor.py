"""The sandbox the engine is allowed to run generated SQL in.

The guard decides *whether* a statement may run; this module decides *how*, and
it assumes the guard may one day have a bug.  Three independent limits apply:

* **Read-only at the driver level** — the file is opened through the
  ``file:...?mode=ro`` URI, so a write reaching this far fails inside SQLite
  with ``attempt to write a readonly database`` rather than mutating a core
  table.  ``Connection.execute`` also refuses stacked statements, which is a
  second line of defence against the statement-splitting bypass.
* **Wall-clock timeout** — enforced with ``set_progress_handler``.  A
  ``sleep``-based watchdog cannot stop a running SQLite statement, and
  ``interrupt()`` needs another thread; the progress handler runs *inside* the
  VM loop, which is the only place a runaway cartesian product can be stopped.
  CPython discards exceptions raised inside a progress callback, so the
  callback records the deadline breach before raising :class:`QueryTimeout`,
  and the wrapper turns the resulting abort into a Korean timeout message.
* **Row cap** — one row beyond the cap is fetched to detect truncation without
  materialising the rest of the result.

Concurrency model: **one connection per thread** (``threading.local``).  No
connection is ever shared, so no lock is needed on the hot path; the only lock
guards the registry that lets :meth:`SQLExecutor.close` reclaim connections
opened by other threads.  This keeps SQLite's per-connection page cache warm
under FastAPI's thread pool, which a connection-per-call design throws away.
"""

from __future__ import annotations

import contextlib
import sqlite3
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from aegis_sql.observability.logging import get_logger
from aegis_sql.types import ExecutionResult

log = get_logger("verify.executor")

#: VM instructions between progress-handler invocations.  Small enough that a
#: runaway join is stopped within a few milliseconds, large enough that the
#: callback overhead stays far below 1% on normal queries.
_PROGRESS_INSTRUCTIONS = 10_000


class QueryTimeout(TimeoutError):
    """Raised when a statement exceeds the executor's wall-clock budget."""


class SQLExecutor:
    """Read-only, time-boxed, row-capped SQLite execution."""

    def __init__(
        self,
        db_path: str | Path,
        timeout_s: float = 8.0,
        max_rows: int = 500,
    ) -> None:
        self.db_path = Path(db_path)
        self.timeout_s = float(timeout_s)
        self.max_rows = int(max_rows)
        self._uri = f"file:{self.db_path}?mode=ro"
        self._local = threading.local()
        self._lock = threading.Lock()
        self._connections: list[sqlite3.Connection] = []

    # -- connection management -------------------------------------------- #

    def _connection(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            return conn
        if not self.db_path.exists():
            raise FileNotFoundError(f"database not found: {self.db_path}")
        # check_same_thread=False only so that close() can reclaim connections
        # owned by worker threads; each connection is still used by one thread.
        conn = sqlite3.connect(self._uri, uri=True, check_same_thread=False)
        conn.execute("PRAGMA query_only = ON")
        self._local.conn = conn
        with self._lock:
            self._connections.append(conn)
        return conn

    def close(self) -> None:
        with self._lock:
            connections, self._connections = self._connections, []
        for conn in connections:
            with contextlib.suppress(sqlite3.Error):  # pragma: no cover - already closed
                conn.close()
        self._local = threading.local()

    # -- execution --------------------------------------------------------- #

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> ExecutionResult:
        """Run ``sql`` and return rows.  Never raises — failures become ``ok=False``."""
        return self._run(sql, params, label="execute")

    def explain(self, sql: str) -> ExecutionResult:
        """``EXPLAIN QUERY PLAN`` — validates identifiers and shows the access path.

        The dry run the pipeline uses before spending real I/O: it resolves every
        table and column without reading a single data page.
        """
        return self._run(f"EXPLAIN QUERY PLAN {sql}", None, label="explain")

    def _run(
        self, sql: str, params: Sequence[Any] | None, label: str
    ) -> ExecutionResult:
        started = time.perf_counter()
        state = {"timed_out": False}
        deadline = started + self.timeout_s

        def progress() -> int:
            if time.perf_counter() >= deadline:
                # The flag is what actually stops the statement: CPython drops
                # exceptions raised inside a progress callback and turns a
                # non-zero return into OperationalError("interrupted").
                state["timed_out"] = True
                raise QueryTimeout(f"{self.timeout_s:g}s budget exceeded")
            return 0

        conn: sqlite3.Connection | None = None
        try:
            conn = self._connection()
            conn.set_progress_handler(progress, _PROGRESS_INSTRUCTIONS)
            cursor = conn.execute(sql, tuple(params or ()))
            columns = [d[0] for d in cursor.description] if cursor.description else []
            fetched = cursor.fetchmany(self.max_rows + 1)
            cursor.close()
            truncated = len(fetched) > self.max_rows
            rows = [tuple(r) for r in fetched[: self.max_rows]]
            elapsed = (time.perf_counter() - started) * 1000.0
            if truncated:
                log.debug("result truncated", rows=self.max_rows, label=label)
            return ExecutionResult(
                ok=True,
                columns=columns,
                rows=rows,
                row_count=len(rows),
                elapsed_ms=elapsed,
                truncated=truncated,
            )
        except Exception as exc:  # noqa: BLE001 - the boundary must not leak
            elapsed = (time.perf_counter() - started) * 1000.0
            error = (
                f"쿼리 시간 초과: {self.timeout_s:g}초 안에 끝나지 않았습니다."
                if state["timed_out"]
                else str(exc)
            )
            log.warning("execution failed", label=label, error=error[:200],
                        elapsed_ms=round(elapsed, 2))
            return ExecutionResult(ok=False, error=error, elapsed_ms=elapsed)
        finally:
            if conn is not None:
                conn.set_progress_handler(None, 0)

    # -- convenience -------------------------------------------------------- #

    def __enter__(self) -> SQLExecutor:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
