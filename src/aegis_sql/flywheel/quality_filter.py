"""Execution-based curation of synthesised pairs — and a leakage-safe split.

Generating training data is easy; the hard part is throwing most of it away for
the right reasons.  Every stage below removes a class of pair that would train
the model to do something wrong, and each is counted so the manifest can show
what the corpus cost:

1. **Executable.**  A pair whose SQL errors is not a hard example, it is a
   broken label.  Everything runs against the real database first.
2. **Non-empty.**  ``WHERE CTRT_STAT_CD = '09'`` runs fine and returns nothing.
   Trained on, it teaches the model that plausible-looking predicates are
   acceptable, which is the single most expensive failure mode in production
   because it is invisible: the query succeeds and the analyst reads "0".
3. **Non-degenerate.**  A single ``NULL``, a ``COUNT`` of zero, or a column of
   one repeated constant carries no signal about whether the SQL was right.
4. **De-duplicated** on ``(sql_skeleton, masked_question)`` with a char-3gram
   Jaccard threshold, so a paraphrase that changed nothing does not occupy a
   slot a genuinely different question could have had.
5. **Difficulty-balanced** to the configured mix by down-sampling, preferring
   pairs that cover an unused skeleton before taking a second from one already
   represented.
6. **Split without leakage** — see below.

Why the split is the interesting stage
--------------------------------------
This is the detail most synthetic-data pipelines get wrong, and it is invisible
until the numbers are already published.  A synthetic corpus contains many pairs
per *query shape*: the same template with a different date window, a different
code value, a different measure.  Shuffle those rows and split 80/10/10 and the
test set is full of near-copies of training rows — same skeleton, same joins,
same clause structure, different constants.  Execution accuracy then measures
memorisation of the generator's grammar and reports it as generalisation, and
the number is inflated by tens of points with nothing in the code looking wrong.

So the unit of assignment here is the **skeleton cluster**, not the row.  Pairs
are grouped by :func:`~aegis_sql.generation.skeleton.sql_skeleton` and whole
clusters go to one split, which guarantees no query shape appears on both sides
of the evaluation boundary.  Augmented paraphrases inherit their parent's SQL,
so they land in the same cluster automatically and cannot leak either.  The
honest cost is that the test set becomes a *compositional* generalisation test
over few shapes — harder, higher-variance, and the number it produces is real.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from aegis_sql.config import Settings
from aegis_sql.observability.logging import get_logger
from aegis_sql.types import ExecutionResult

log = get_logger("flywheel.quality_filter")

__all__ = ["Pair", "QualityFilter", "SPLITS"]

SPLITS: tuple[str, ...] = ("train", "dev", "test")
_SPLIT_RATIO: dict[str, float] = {"train": 0.8, "dev": 0.1, "test": 0.1}

try:  # The generation package owns these; the flywheel must not depend on load order.
    from aegis_sql.generation.skeleton import mask_question, normalize_sql, sql_skeleton

    _NATIVE_SKELETON = True
except Exception:  # pragma: no cover - exercised only before generation/ lands
    _NATIVE_SKELETON = False

    _TOKEN_RE = re.compile(
        r"'(?:[^']|'')*'|\d+(?:\.\d+)?|[A-Za-z_][A-Za-z0-9_.]*|<=|>=|<>|!=|\|\||[(),;.*=<>+\-/%]"
    )
    _KEYWORDS = frozenset(
        [
            "select", "from", "where", "group", "by", "order", "having", "limit", "join",
            "left", "right", "inner", "outer", "on", "and", "or", "not", "in", "exists",
            "between", "like", "is", "null", "as", "union", "all", "distinct", "case", "when",
            "then", "else", "end", "with", "count", "sum", "avg", "min", "max", "cast", "real",
            "integer", "nullif", "substr", "round", "desc", "asc",
        ]
    )

    def normalize_sql(sql: str, dialect: str = "sqlite") -> str:
        return " ".join((sql or "").lower().split())

    def sql_skeleton(sql: str, dialect: str = "sqlite") -> str:
        out: list[str] = []
        for token in _TOKEN_RE.findall(sql or ""):
            if token.lower() in _KEYWORDS:
                out.append(token.lower())
            elif token.startswith("'") or token[0].isdigit():
                out.append("?")
            elif token[0].isalpha() or token[0] == "_":
                out.append("_")
            else:
                out.append(token)
        return " ".join(out)

    def mask_question(text: str) -> str:
        return re.sub(r"\d[\d,]*", "<NUM>", (text or "").strip())


@runtime_checkable
class Executable(Protocol):
    """The slice of :class:`~aegis_sql.verify.executor.SQLExecutor` used here."""

    def execute(self, sql: str, params: Any = None) -> ExecutionResult: ...


@dataclass(slots=True)
class Pair:
    """One (question, SQL) training example plus its provenance."""

    question: str
    sql: str
    difficulty: str
    template_id: str
    source: str = "backtranslate"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def split(self) -> str:
        return str(self.meta.get("split", ""))


class QualityFilter:
    """Runs the curation stages and assigns leakage-safe splits."""

    def __init__(self, executor: Executable, settings: Settings) -> None:
        self.executor = executor
        self.settings = settings
        self.require_nonempty = bool(settings.flywheel.require_nonempty_result)
        self.dedupe_threshold = float(settings.flywheel.dedupe_threshold)
        self.mix = dict(settings.flywheel.difficulty_mix)
        #: Augmented siblings share their parent's SQL, so the database is hit
        #: once per distinct statement rather than once per pair.
        self._results: dict[str, ExecutionResult] = {}
        self._cache_hits = 0

    # -- public ------------------------------------------------------------- #

    def filter(self, pairs: list[Pair]) -> tuple[list[Pair], dict[str, Any]]:
        stats: dict[str, Any] = {
            "input": len(pairs),
            "stages": {},
            "drop_reasons": Counter(),
            "exec_errors": Counter(),
        }
        survivors = self._stage_execute(pairs, stats)
        survivors = self._stage_degenerate(survivors, stats)
        survivors = self._stage_dedupe(survivors, stats)
        survivors = self._stage_balance(survivors, stats)
        survivors = self._stage_split(survivors, stats)

        stats["kept"] = len(survivors)
        stats["difficulty"] = dict(Counter(p.difficulty for p in survivors))
        stats["templates"] = dict(Counter(p.template_id for p in survivors))
        stats["drop_reasons"] = dict(stats["drop_reasons"])
        stats["exec_errors"] = dict(Counter(stats["exec_errors"]).most_common(5))
        stats["executions"] = {
            "unique_sql": len(self._results),
            "cache_hits": self._cache_hits,
            "skeleton_impl": "native" if _NATIVE_SKELETON else "fallback",
        }
        log.info(
            "quality filter complete",
            input=stats["input"],
            kept=stats["kept"],
            unique_sql=len(self._results),
            splits=stats.get("split"),
        )
        return survivors, stats

    def execution_of(self, sql: str) -> ExecutionResult:
        """Memoised execution — public so a caller can reuse the same evidence."""
        key = normalize_sql(sql)
        cached = self._results.get(key)
        if cached is not None:
            self._cache_hits += 1
            return cached
        result = self.executor.execute(sql)
        self._results[key] = result
        return result

    # -- stages -------------------------------------------------------------- #

    def _stage_execute(self, pairs: list[Pair], stats: dict[str, Any]) -> list[Pair]:
        """Stages 1 and 2: the SQL must run, and (optionally) return rows."""
        kept: list[Pair] = []
        empty = 0
        for pair in pairs:
            result = self.execution_of(pair.sql)
            if not result.ok:
                stats["drop_reasons"]["exec_error"] += 1
                stats["exec_errors"][_error_key(result.error)] += 1
                continue
            if self.require_nonempty and result.row_count == 0:
                stats["drop_reasons"]["empty_result"] += 1
                empty += 1
                continue
            pair.meta.setdefault("row_count", result.row_count)
            pair.meta.setdefault("result_signature", result.result_signature())
            kept.append(pair)
        stats["stages"]["executable"] = {
            "in": len(pairs),
            "out": len(kept),
            "dropped_error": stats["drop_reasons"].get("exec_error", 0),
            "dropped_empty": empty,
        }
        return kept

    def _stage_degenerate(self, pairs: list[Pair], stats: dict[str, Any]) -> list[Pair]:
        """Stage 3: drop results that cannot distinguish right SQL from wrong SQL."""
        kept: list[Pair] = []
        for pair in pairs:
            reason = _degenerate_reason(self.execution_of(pair.sql))
            if reason:
                stats["drop_reasons"][reason] += 1
                continue
            kept.append(pair)
        stats["stages"]["degenerate"] = {"in": len(pairs), "out": len(kept)}
        return kept

    def _stage_dedupe(self, pairs: list[Pair], stats: dict[str, Any]) -> list[Pair]:
        """Stage 4: near-duplicate questions within one query shape are redundant."""
        kept: list[Pair] = []
        seen_exact: set[tuple[str, str]] = set()
        by_skeleton: dict[str, list[set[str]]] = defaultdict(list)
        for pair in pairs:
            skeleton = _skeleton_of(pair)
            masked = mask_question(pair.question)
            exact = (skeleton, masked)
            if exact in seen_exact:
                stats["drop_reasons"]["duplicate_exact"] += 1
                continue
            grams = _trigrams(masked)
            if any(_jaccard(grams, other) >= self.dedupe_threshold for other in by_skeleton[skeleton]):
                stats["drop_reasons"]["duplicate_near"] += 1
                continue
            seen_exact.add(exact)
            by_skeleton[skeleton].append(grams)
            pair.meta["sql_skeleton"] = skeleton
            pair.meta["masked_question"] = masked
            kept.append(pair)
        stats["stages"]["dedupe"] = {
            "in": len(pairs),
            "out": len(kept),
            "skeletons": len(by_skeleton),
        }
        return kept

    def _stage_balance(self, pairs: list[Pair], stats: dict[str, Any]) -> list[Pair]:
        """Stage 5: down-sample to the configured difficulty mix, coverage first."""
        counts = Counter(p.difficulty for p in pairs)
        weights = {d: w for d, w in self.mix.items() if w > 0 and counts.get(d, 0) > 0}
        if not weights:
            stats["stages"]["balance"] = {"in": len(pairs), "out": len(pairs), "skipped": True}
            return pairs
        missing = [d for d, w in self.mix.items() if w > 0 and counts.get(d, 0) == 0]
        if missing:
            log.warning("difficulty absent from corpus; mix renormalised", missing=missing)
        total_weight = sum(weights.values())
        scale = min(counts[d] * total_weight / w for d, w in weights.items())
        targets = {d: int(round(w * scale / total_weight)) for d, w in weights.items()}

        taken: Counter[str] = Counter()
        selected: list[tuple[int, Pair]] = []
        for _key, index, pair in sorted(_coverage_ranking(pairs)):
            if taken[pair.difficulty] >= targets.get(pair.difficulty, 0):
                stats["drop_reasons"]["difficulty_balance"] += 1
                continue
            taken[pair.difficulty] += 1
            selected.append((index, pair))
        # Restore corpus order; the ranking was only a selection device.
        kept = [pair for _index, pair in sorted(selected, key=lambda item: item[0])]
        stats["stages"]["balance"] = {"in": len(pairs), "out": len(kept), "targets": targets}
        return kept

    def _stage_split(self, pairs: list[Pair], stats: dict[str, Any]) -> list[Pair]:
        """Stage 6: assign whole skeleton clusters so no shape spans two splits."""
        clusters: dict[str, list[Pair]] = defaultdict(list)
        for pair in pairs:
            clusters[pair.meta.get("sql_skeleton") or _skeleton_of(pair)].append(pair)

        assigned: Counter[str] = Counter()
        cluster_counts: Counter[str] = Counter()
        # Assignment runs *per difficulty*.  A cluster is one query shape and so
        # carries one difficulty; balancing globally would hand a whole bucket to
        # one split and leave the test set with no hard examples at all.
        by_difficulty: dict[str, list[str]] = defaultdict(list)
        for skeleton, members in clusters.items():
            by_difficulty[Counter(p.difficulty for p in members).most_common(1)[0][0]].append(skeleton)

        for difficulty in sorted(by_difficulty):
            skeletons = by_difficulty[difficulty]
            total = sum(len(clusters[s]) for s in skeletons)
            local: Counter[str] = Counter()
            # Largest clusters first: placing the big rocks before the sand is
            # what keeps a 10% split from overshooting to 25% on the last one.
            for skeleton in sorted(skeletons, key=lambda s: (-len(clusters[s]), s)):
                members = clusters[skeleton]
                split = max(
                    SPLITS, key=lambda s: (_SPLIT_RATIO[s] * total - local[s], -SPLITS.index(s))
                )
                local[split] += len(members)
                assigned[split] += len(members)
                cluster_counts[split] += 1
                for pair in members:
                    pair.meta["split"] = split

        stats["split"] = {s: assigned.get(s, 0) for s in SPLITS}
        stats["clusters"] = {"total": len(clusters), **{s: cluster_counts.get(s, 0) for s in SPLITS}}
        stats["stages"]["split"] = {"in": len(pairs), "out": len(pairs), "clusters": len(clusters)}
        _assert_no_leakage(pairs)
        return pairs


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _skeleton_of(pair: Pair) -> str:
    cached = pair.meta.get("sql_skeleton")
    if cached:
        return str(cached)
    return sql_skeleton(pair.sql)


def _trigrams(text: str) -> set[str]:
    squished = re.sub(r"\s+", "", text)
    if len(squished) < 3:
        return {squished} if squished else set()
    return {squished[i : i + 3] for i in range(len(squished) - 2)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0


def _error_key(error: str | None) -> str:
    """Collapse an error message to its class so the counter stays readable."""
    text = (error or "unknown").strip().lower()
    return re.sub(r"[\"'][^\"']*[\"']", "?", text)[:60]


def _degenerate_reason(result: ExecutionResult) -> str | None:
    """Name the way a result set fails to be informative, or ``None``."""
    if not result.ok:
        return None  # the executable stage already removed these
    if result.row_count == 0:
        # Only reachable with ``require_nonempty_result: false``; an empty result
        # is still the least informative outcome there is.
        return "degenerate_empty"
    rows, columns = result.rows, result.columns
    if len(columns) == 1 and result.row_count == 1:
        value = rows[0][0]
        if value is None:
            return "degenerate_null"
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0:
            return "degenerate_zero"
        if isinstance(value, str) and not value.strip():
            return "degenerate_blank"
    if all(all(cell is None for cell in row) for row in rows):
        return "degenerate_all_null"
    if len(columns) == 1 and result.row_count > 1 and len({row[0] for row in rows}) == 1:
        return "degenerate_constant"
    return None


def _coverage_ranking(pairs: list[Pair]) -> list[tuple[tuple[int, str], int, Pair]]:
    """Rank pairs so the first take from every skeleton precedes any second take.

    Under a down-sampling budget this maximises the number of distinct query
    shapes that survive, which is the axis a training corpus is actually short
    of — a second date window on a shape already present adds almost nothing.
    The secondary key is a content hash rather than ``hash()``, which is salted
    per interpreter and would make the corpus differ between runs.
    """
    occurrence: Counter[str] = Counter()
    out: list[tuple[tuple[int, str], int, Pair]] = []
    for index, pair in enumerate(pairs):
        skeleton = str(pair.meta.get("sql_skeleton") or "")
        occurrence[skeleton] += 1
        key = (occurrence[skeleton], _stable_hash(f"{pair.question}\n{pair.sql}"))
        out.append((key, index, pair))
    return out


def _stable_hash(text: str) -> str:
    """Process-independent ordering key (``hash()`` is salted per interpreter)."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()


def _assert_no_leakage(pairs: list[Pair]) -> None:
    """Invariant check: one skeleton, one split.  Cheap, and worth failing loudly."""
    owner: dict[str, str] = {}
    for pair in pairs:
        skeleton = str(pair.meta.get("sql_skeleton") or "")
        split = pair.split
        if owner.setdefault(skeleton, split) != split:
            raise AssertionError(f"skeleton leaked across splits: {skeleton[:80]!r}")
