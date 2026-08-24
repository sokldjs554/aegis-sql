"""Voting on what the SQL *returns*, not on how it is written.

Sampling a model several times and keeping the majority string is the standard
self-consistency recipe, and it under-counts badly on SQL: ``JOIN`` order,
alias names, ``IN`` versus ``OR``, ``COUNT(*)`` versus ``COUNT(1)`` all produce
different strings with identical semantics, so string voting shatters one
correct answer into five singleton groups.  Voting on
:meth:`~aegis_sql.types.ExecutionResult.result_signature` — an order-insensitive
hash of the result set — puts those five back in one group, which is the whole
point of executing candidates in the first place.

Two details matter in practice:

* **Empty results do not win by default.**  A wrong predicate returns zero rows
  and is trivially reproducible, so several broken candidates agree with each
  other.  A group with rows outranks any empty group; only when nothing returns
  rows does an empty group win.
* **Duplicates vote but execute once.**  Identical statements are the strongest
  agreement signal there is, so they keep their votes, but re-running them
  would just burn the query budget.

The resulting agreement ratio doubles as a calibration signal for the router:
low agreement is the cheapest available predictor that the question was harder
than the tier that answered it.
"""

from __future__ import annotations

from typing import Any, Protocol

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

from aegis_sql.observability.logging import get_logger
from aegis_sql.types import ExecutionResult, SQLCandidate

log = get_logger("verify.selfconsistency")

DIALECT = "sqlite"


class _Executor(Protocol):
    def execute(self, sql: str, params: Any = None) -> ExecutionResult: ...


class _Guard(Protocol):
    def check(self, sql: str, ctx: dict | None = None) -> Any: ...


def vote(
    candidates: list[SQLCandidate],
    executor: _Executor,
    guard: _Guard | None = None,
) -> tuple[SQLCandidate | None, dict]:
    """Execute the candidates, group them by result, and elect the largest group.

    Every candidate comes back annotated with ``valid``/``error``/``votes``.
    Returns ``(None, stats)`` when no candidate produced a result set — the
    caller keeps its own ordering and falls through to the repair loop.
    """
    stats = {"groups": 0, "agreement": 0.0, "executed": 0}
    if not candidates:
        return None, stats

    representative: dict[str, int] = {}
    members: dict[int, list[int]] = {}
    results: dict[int, ExecutionResult] = {}

    for index, candidate in enumerate(candidates):
        key = candidate.normalized_key()
        if key in representative:
            members[representative[key]].append(index)
            continue
        representative[key] = index
        members[index] = [index]
        results[index] = _run(candidate, executor, guard)
        stats["executed"] += 1

    groups: dict[str, list[int]] = {}
    for index, result in results.items():
        if not result.ok:
            continue
        groups.setdefault(result.result_signature(), []).append(index)

    for index, result in results.items():
        votes = (
            sum(len(members[i]) for i in groups[result.result_signature()]) if result.ok else 0
        )
        for member in members[index]:
            candidates[member].valid = result.ok
            candidates[member].error = None if result.ok else result.error
            candidates[member].votes = votes

    stats["groups"] = len(groups)
    if not groups:
        log.warning("self-consistency: no candidate executed", candidates=len(candidates))
        return None, stats

    non_empty = {
        sig: idx for sig, idx in groups.items() if any(results[i].row_count > 0 for i in idx)
    }
    pool = non_empty or groups

    def group_rank(signature: str) -> tuple[int, int, int, int]:
        indices = pool[signature]
        votes = sum(len(members[i]) for i in indices)
        best = min(indices, key=lambda i: _candidate_rank(candidates[i], i))
        joins, length, order = _candidate_rank(candidates[best], best)
        return (-votes, joins, length, order)

    winning_signature = min(pool, key=group_rank)
    winning_indices = pool[winning_signature]
    winner_index = min(winning_indices, key=lambda i: _candidate_rank(candidates[i], i))
    winner = candidates[winner_index]

    total = len(candidates)
    winner_votes = sum(len(members[i]) for i in winning_indices)
    stats["agreement"] = round(winner_votes / total, 4) if total else 0.0
    log.info(
        "self-consistency vote",
        candidates=total,
        executed=stats["executed"],
        groups=stats["groups"],
        agreement=stats["agreement"],
    )
    return winner, stats


def agreement_score(candidates: list[SQLCandidate]) -> float:
    """Fraction of candidates that agree with the plurality answer.

    Uses execution votes when :func:`vote` has already run, and falls back to
    normalised-string grouping so the score is still defined pre-execution.
    """
    if not candidates:
        return 0.0
    votes = [c.votes for c in candidates if c.votes]
    if votes:
        return round(max(votes) / len(candidates), 4)
    buckets: dict[str, int] = {}
    for candidate in candidates:
        key = candidate.normalized_key()
        buckets[key] = buckets.get(key, 0) + 1
    return round(max(buckets.values()) / len(candidates), 4)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _run(candidate: SQLCandidate, executor: _Executor, guard: _Guard | None) -> ExecutionResult:
    """Guard first when a guard is supplied — a candidate is untrusted input too."""
    sql = candidate.sql
    if guard is not None:
        verdict = guard.check(sql)
        if not verdict.allowed:
            codes = ", ".join(v.code for v in verdict.violations if v.severity == "block")
            return ExecutionResult(ok=False, error=f"정책 위반으로 후보에서 제외: {codes}")
        sql = verdict.rewritten_sql or sql
    return executor.execute(sql)


def _candidate_rank(candidate: SQLCandidate, index: int) -> tuple[int, int, int]:
    """Tie-break key: fewer joins, then shorter SQL, then earlier in the sample order."""
    return (_join_count(candidate.sql), len(candidate.sql), index)


def _join_count(sql: str) -> int:
    try:
        root = sqlglot.parse_one(sql, dialect=DIALECT)
    except (ParseError, TokenError, RecursionError):
        return sql.lower().count(" join ")
    if root is None:
        return 0
    explicit = len(list(root.find_all(exp.Join)))
    subqueries = len(list(root.find_all(exp.Subquery)))
    return explicit + subqueries
