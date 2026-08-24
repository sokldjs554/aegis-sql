"""Schema-grounded SQL program sampler — the source of the training corpus.

There is no public Korean insurance Text-to-SQL dataset, and there will never be
one: the schemas are proprietary and the questions are internal.  So the corpus
has to be manufactured, and the only trustworthy starting point is the database
itself.  This module samples *SQL first* and lets the natural-language side be
derived from it (``back_translate.py``), which inverts the usual annotation
order for one decisive reason: a sampled program is executable by construction,
so every pair in the corpus can be verified against real data before a model
ever sees it.  Annotating questions first gives you SQL nobody can check.

What "grounded" means here
--------------------------
A grammar that fills slots at random produces syntactically valid SQL that is
semantically dead — ``WHERE CTRT_STAT_CD = 'Z9'`` parses, executes, and returns
zero rows forever.  Every slot here is therefore drawn from the *observed* data
in :class:`~aegis_sql.schema.profile.SchemaProfile`:

* **Code values** come from the resolved ``TB_COMM_CD`` labels of that exact
  column, so the predicate always names a code the column actually holds.
* **Date windows** are intersected with the column's observed ``MIN``/``MAX``
  and rejected below ~45 days of overlap, which is what stops the sampler
  asking about 2026-12 when the extract ends in August.
* **Thresholds** are order statistics over the column's frequency-ranked
  values.  The profile keeps the 25 most frequent values, so this is a quantile
  of the *modal band* rather than of the full distribution — deliberately, since
  a true 90th percentile on a long-tailed KRW column selects a handful of rows
  and teaches the model a predicate shape it will never be asked for.
* **Joins** are FK paths taken from :class:`~aegis_sql.schema.graph.JoinGraph`,
  and aggregation templates only follow edges in the many→one direction so that
  ``SUM(premium)`` is not silently multiplied by a one-to-many fan-out.

Column roles (measure / date / code / categorical / label / key) are *inferred*
from the profile and the data dictionary rather than hard-coded, so pointing the
sampler at another Korean core produces a corpus for that core.  Columns whose
Korean comment marks them as personal data (주민등록번호, 연락처, 주소, …) never
reach a template: a synthetic corpus that teaches a model to SELECT a phone
number is a governance incident waiting to be trained in.

Difficulty buckets are the *curriculum* label, defined by the reasoning a human
analyst needs, and are intentionally not identical to
:func:`~aegis_sql.generation.skeleton.difficulty_of`, which grades the emitted
SQL structurally.  ``build_dataset`` reports both distributions so the gap stays
visible instead of being quietly averaged away.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from aegis_sql.observability.logging import get_logger
from aegis_sql.schema.graph import JoinEdge, JoinGraph
from aegis_sql.schema.introspect import CODE_NAME_COLUMN
from aegis_sql.schema.profile import ColumnProfile, SchemaProfile
from aegis_sql.types import ColumnInfo, GlossaryEntry, SchemaGraph, Sensitivity

log = get_logger("flywheel.sql_sampler")

__all__ = ["SQLProgram", "SQLSampler", "template_ids", "DEFAULT_MIX"]

DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")

#: Fallback when the caller passes an empty or unusable mix.
DEFAULT_MIX: dict[str, float] = {"easy": 0.30, "medium": 0.45, "hard": 0.25}

_NUMERIC_TYPES = frozenset(
    {"INTEGER", "INT", "REAL", "NUMERIC", "DECIMAL", "FLOAT", "BIGINT", "DOUBLE", "SMALLINT"}
)

#: Physical-name markers for personal data.  Applied together with the Korean
#: comment lexicon below, because either one alone has blind spots.
_PII_NAME_RE = re.compile(
    r"(RRNO|SSN|JUMIN|TELNO|PHONE|MOBILE|EMAIL|ADDR|ZIP|POST_?CD|LIC_?NO|ACCT_?NO|PASSWD|PWD)",
    re.IGNORECASE,
)

#: The data dictionary is the better signal: it says what the column *means*.
_PII_COMMENT_RE = re.compile(
    r"주민등록번호|주민번호|연락처|전화|휴대폰|이메일|상세주소|주소|우편번호|계좌번호|비밀번호|"
    r"성명|고객명|생년월일|자격증번호"
)

#: Measures whose Korean label marks them as intensive (a score, a rate, a
#: duration): summing them is meaningless, so only AVG/MAX/MIN are offered.
_NON_ADDITIVE_RE = re.compile(r"점수|비율|율$|지수|등급|일수|년수|연령|나이|만족도|순서")

#: Health/opinion columns: aggregatable, never projected or grouped row-by-row.
_RESTRICTED_COMMENT_RE = re.compile(r"진단|병원|질병|상병|심사의견|상담내용|이상징후")

#: Below this, an aggregate over the column is not worth asking about
#: (``RVIV_CNT`` has two distinct values; ``AVG`` of it is a trick question).
_MIN_MEASURE_DISTINCT = 5
#: Free text masquerading as a categorical (상담내용) is excluded by length.
_MAX_CATEGORICAL_LENGTH = 20.0
#: A table needs this many rows before it is worth aggregating over.
_MIN_SOURCE_ROWS = 100
#: A sampled date window must overlap the observed range by at least this much.
_MIN_WINDOW_OVERLAP_DAYS = 45
#: Rejection sampling budget per requested program.
_MAX_ATTEMPTS = 30


@dataclass(slots=True)
class SQLProgram:
    """One sampled statement plus the slot bindings that produced it.

    ``slots`` is the interface to :mod:`aegis_sql.flywheel.back_translate`: it
    carries *physical* references (qualified columns, code values, date bounds)
    and the *shape* of each binding (``date_kind``, ``agg``, ``op``), never
    pre-rendered Korean.  Verbalisation is the translator's job, which is what
    lets the same program be realised many different ways.
    """

    sql: str
    difficulty: str
    tables: list[str]
    template_id: str
    slots: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Whitespace/case-insensitive identity, used to de-duplicate samples."""
        return " ".join(self.sql.lower().split())


@dataclass(slots=True)
class _Roles:
    """Column roles inferred for one table — the sampler's view of a schema."""

    table: str
    measures: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    codes: list[str] = field(default_factory=list)
    categoricals: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    fk_columns: list[str] = field(default_factory=list)
    key: str | None = None

    @property
    def dimensions(self) -> list[str]:
        """Columns that make a sensible ``GROUP BY`` axis."""
        return [*self.codes, *self.categoricals, *self.labels]


@dataclass(slots=True)
class _Window:
    """A date window sampled inside a column's observed range."""

    column: str  # qualified "TBL.COL"
    start: str  # YYYYMMDD
    end: str  # YYYYMMDD
    kind: str  # year | half | quarter | month | recent | range
    year: int = 0
    part: int = 0  # half 1-2, quarter 1-4, month 1-12
    n_months: int = 0

    def slots(self, prefix: str = "date") -> dict[str, Any]:
        return {
            f"{prefix}_col": self.column,
            f"{prefix}_start": self.start,
            f"{prefix}_end": self.end,
            f"{prefix}_kind": self.kind,
            f"{prefix}_year": self.year,
            f"{prefix}_part": self.part,
            f"{prefix}_n": self.n_months,
        }


_TemplateFn = Callable[["SQLSampler", random.Random], "SQLProgram | None"]


@dataclass(slots=True)
class _TemplateSpec:
    template_id: str
    difficulty: str
    build: _TemplateFn


#: Registry populated by the ``@_template`` decorators in the class body below.
_TEMPLATES: dict[str, _TemplateSpec] = {}


def _template(template_id: str, difficulty: str) -> Callable[[_TemplateFn], _TemplateFn]:
    def register(fn: _TemplateFn) -> _TemplateFn:
        _TEMPLATES[template_id] = _TemplateSpec(template_id, difficulty, fn)
        return fn

    return register


def template_ids(difficulty: str | None = None) -> list[str]:
    """Every registered template id, optionally filtered to one bucket."""
    return [t for t, spec in _TEMPLATES.items() if difficulty is None or spec.difficulty == difficulty]


# --------------------------------------------------------------------------- #
# sampler
# --------------------------------------------------------------------------- #


class SQLSampler:
    """Draws executable, schema-grounded SQL programs from a seeded grammar."""

    def __init__(
        self,
        schema: SchemaGraph,
        profile: SchemaProfile,
        join_graph: JoinGraph,
        glossary: Any = None,
        seed: int = 20260824,
    ) -> None:
        self.schema = schema
        self.profile = profile
        self.join_graph = join_graph
        self.glossary: list[GlossaryEntry] = _as_entries(glossary)
        self.seed = int(seed)

        self.code_table: str | None = next(
            (t for t in ("TB_COMM_CD", "TB_CODE", "TC_CMMN_CD", "COMMON_CODE") if schema.table(t)), None
        )
        self._roles: dict[str, _Roles] = {t: self._infer_roles(t) for t in sorted(schema.tables)}
        #: Descending (many→one) FK edges, keyed so aggregation never fans out.
        self._descending: set[tuple[str, str, str, str]] = {
            (fk.from_table, fk.from_column, fk.to_table, fk.to_column) for fk in schema.foreign_keys
        }
        self._edges: dict[str, list[JoinEdge]] = {
            t: self._neighbour_edges(t) for t in sorted(schema.tables)
        }
        self.sources: list[str] = self._pick_sources()
        log.info(
            "sql sampler ready",
            templates=len(_TEMPLATES),
            sources=len(self.sources),
            code_table=self.code_table or "-",
        )

    # -- public ----------------------------------------------------------- #

    def sample(self, n: int, mix: dict[str, float] | None = None) -> list[SQLProgram]:
        """Draw ``n`` distinct programs with the requested difficulty mix.

        Rejection sampling: a template that cannot bind its slots for the table
        it drew returns ``None`` and the draw is retried.  Buckets that run out
        of distinct programs come up short rather than emitting duplicates — a
        corpus with 400 copies of one statement is worse than a smaller one.
        """
        rng = random.Random(self.seed)
        targets = _allocate(n, mix or DEFAULT_MIX)
        seen: set[str] = set()
        out: list[SQLProgram] = []

        for difficulty in DIFFICULTIES:
            want = targets.get(difficulty, 0)
            specs = [s for s in _TEMPLATES.values() if s.difficulty == difficulty]
            if want <= 0 or not specs:
                continue
            produced, attempts, budget = 0, 0, want * _MAX_ATTEMPTS
            # Round-robin over a reshuffled cycle rather than i.i.d. draws: a
            # template with tight slot requirements (a 3-table chain plus an
            # out-of-tree subquery) would otherwise lose every race against the
            # permissive ones and never appear in the corpus at all.
            cycle: list[_TemplateSpec] = []
            while produced < want and attempts < budget:
                if not cycle:
                    cycle = list(specs)
                    rng.shuffle(cycle)
                attempts += 1
                program = cycle.pop().build(self, rng)
                if program is None or program.key in seen:
                    continue
                seen.add(program.key)
                out.append(program)
                produced += 1
            if produced < want:
                log.warning(
                    "difficulty bucket under-filled",
                    difficulty=difficulty,
                    requested=want,
                    produced=produced,
                )

        rng.shuffle(out)
        log.info(
            "sampled sql programs",
            n=len(out),
            easy=sum(p.difficulty == "easy" for p in out),
            medium=sum(p.difficulty == "medium" for p in out),
            hard=sum(p.difficulty == "hard" for p in out),
            distinct_templates=len({p.template_id for p in out}),
        )
        return out

    def roles(self, table: str) -> _Roles:
        return self._roles.get(table, _Roles(table=table))

    # -- schema role inference -------------------------------------------- #

    def _infer_roles(self, table: str) -> _Roles:
        info = self.schema.table(table)
        roles = _Roles(table=table)
        if info is None or table == self.code_table:
            return roles
        pk = info.primary_key
        roles.key = pk[0] if len(pk) == 1 else None

        for col in info.columns:
            prof = self.profile.get(table, col.name)
            if prof is None or _is_pii(col):
                continue
            qualified = col.name
            if col.foreign_key is not None:
                roles.fk_columns.append(qualified)

            if col.dtype.upper() in _NUMERIC_TYPES:
                if (
                    not col.is_primary_key
                    and col.foreign_key is None
                    and prof.distinct_count >= _MIN_MEASURE_DISTINCT
                ):
                    roles.measures.append(qualified)
                continue

            if prof.is_yyyymmdd:
                roles.dates.append(qualified)
            elif col.code_group and prof.code_labels:
                roles.codes.append(qualified)
            elif _is_restricted(col):
                continue
            elif col.name.upper().endswith("_NM") and prof.distinct_count > 1:
                roles.labels.append(qualified)
            elif (
                prof.is_categorical
                and 1 < prof.distinct_count <= 40
                and prof.avg_length <= _MAX_CATEGORICAL_LENGTH
                and not col.is_primary_key
            ):
                roles.categoricals.append(qualified)

        return roles

    def _pick_sources(self) -> list[str]:
        """Tables big enough and rich enough to be the grain of a question."""
        out: list[str] = []
        for name in sorted(self.schema.tables):
            info = self.schema.table(name)
            roles = self._roles[name]
            if info is None or name == self.code_table:
                continue
            if info.row_count >= 0 and info.row_count < _MIN_SOURCE_ROWS:
                continue
            if roles.dates and (roles.measures or roles.codes or roles.fk_columns):
                out.append(name)
        return out

    # -- slot samplers ----------------------------------------------------- #

    def _source(self, rng: random.Random, need_measure: bool = False) -> str | None:
        pool = [t for t in self.sources if not need_measure or self._roles[t].measures]
        return pool[rng.randrange(len(pool))] if pool else None

    def _pick(self, rng: random.Random, options: list[str]) -> str | None:
        return options[rng.randrange(len(options))] if options else None

    def _profile_of(self, qualified: str) -> ColumnProfile | None:
        table, _, column = qualified.partition(".")
        return self.profile.get(table, column)

    def _code_value(self, rng: random.Random, table: str, column: str) -> tuple[str, str] | None:
        """A ``(code, label)`` pair the column actually contains."""
        prof = self.profile.get(table, column)
        if prof is None or not prof.code_labels:
            return None
        present = [v for v in sorted(prof.code_labels) if v in prof.values]
        pool = present or sorted(prof.code_labels)
        code = pool[rng.randrange(len(pool))]
        return code, prof.code_labels[code]

    def _categorical_value(self, rng: random.Random, table: str, column: str) -> str | None:
        prof = self.profile.get(table, column)
        if prof is None or not prof.values:
            return None
        return prof.values[rng.randrange(min(len(prof.values), 12))]

    def _is_additive(self, table: str, column: str) -> bool:
        col = self.schema.column(table, column)
        return not _NON_ADDITIVE_RE.search((col.comment if col else None) or column)

    def _agg_for(self, rng: random.Random, table: str, column: str) -> str:
        """An aggregate that means something for this measure.

        ``SUM(만족도점수)`` type-checks and executes; it is also nonsense, and a
        corpus full of it teaches the model that any aggregate fits any column.
        """
        if self._is_additive(table, column):
            return rng.choice(("SUM", "AVG", "MAX", "MIN"))
        return rng.choice(("AVG", "MAX", "MIN"))

    def _threshold(self, rng: random.Random, table: str, column: str) -> int | None:
        """An order statistic over the observed values, rounded to a Korean unit."""
        prof = self.profile.get(table, column)
        if prof is None:
            return None
        numbers = sorted(int(float(v)) for v in prof.values if _is_number(v))
        if len(numbers) < 3:
            return None
        quantile = rng.choice((0.25, 0.4, 0.5, 0.6, 0.75))
        index = min(len(numbers) - 1, max(0, round(quantile * (len(numbers) - 1))))
        return _round_nice(numbers[index])

    def _window(self, rng: random.Random, table: str, column: str, offset_years: int = 0) -> _Window | None:
        """Pick a natural calendar window that overlaps the column's real range.

        ``offset_years`` shifts the window back, which is how the year-on-year
        template gets two *aligned* windows instead of two unrelated ones.
        """
        prof = self.profile.get(table, column)
        if prof is None or not prof.min_value or not prof.max_value:
            return None
        lo, hi = prof.min_value, prof.max_value
        if len(lo) != 8 or len(hi) != 8 or not (lo.isdigit() and hi.isdigit()):
            return None
        qualified = f"{table}.{column}"

        candidates: list[_Window] = []
        for year in range(int(lo[:4]) + offset_years, int(hi[:4]) + 1):
            base = year - offset_years
            candidates.append(_Window(qualified, f"{base}0101", f"{base}1231", "year", year=base))
            for half in (1, 2):
                start, end = (f"{base}0101", f"{base}0630") if half == 1 else (f"{base}0701", f"{base}1231")
                candidates.append(_Window(qualified, start, end, "half", year=base, part=half))
            for quarter in (1, 2, 3, 4):
                first = 3 * (quarter - 1) + 1
                start = f"{base}{first:02d}01"
                end = _month_end(base, first + 2)
                candidates.append(_Window(qualified, start, end, "quarter", year=base, part=quarter))
            for month in (2, 5, 8, 11):
                candidates.append(
                    _Window(qualified, f"{base}{month:02d}01", _month_end(base, month), "month", base, month)
                )

            for span in (9, 18):
                start = f"{base}0401"
                tail = _shift_months(start, span - 1)
                candidates.append(
                    _Window(qualified, start, _month_end(int(tail[:4]), int(tail[4:6])), "range",
                            year=base, n_months=span)
                )

        # "최근 N개월" is anchored to the last observed date, not to the clock,
        # so the same profile always yields the same window.
        if offset_years == 0:
            for months in (3, 6, 12):
                start = _shift_months(hi, -months)
                candidates.append(_Window(qualified, start, hi, "recent", n_months=months))

        viable = [w for w in candidates if _overlap_days(w.start, w.end, lo, hi) >= _MIN_WINDOW_OVERLAP_DAYS]
        return viable[rng.randrange(len(viable))] if viable else None

    # -- join helpers ------------------------------------------------------ #

    def _neighbour_edges(self, table: str) -> list[JoinEdge]:
        edges: list[JoinEdge] = []
        for neighbour in sorted(self.join_graph.neighbours(table)):
            if neighbour == self.code_table:
                continue
            hop = self.join_graph.shortest_path(table, neighbour)
            if hop:
                edges.append(hop[0])
        return edges

    def _parents(self, table: str) -> list[JoinEdge]:
        """Many→one edges: the dimensions ``table`` can be grouped by."""
        return [e for e in self._edges.get(table, []) if self._descends(e)]

    def _children(self, table: str) -> list[JoinEdge]:
        """One→many edges: the detail tables whose absence an anti-join tests."""
        return [e for e in self._edges.get(table, []) if not self._descends(e)]

    def _source_with(self, rng: random.Random, *requirements: Callable[[str], bool]) -> str | None:
        """Draw a source table that satisfies every requirement.

        Drawing blind and rejecting is fine for a template whose slots almost
        always bind; for the anti-join shapes, only two tables in the schema
        qualify, and blind draws lose the round-robin race often enough that the
        template never reaches the corpus.
        """
        pool = [t for t in self.sources if all(req(t) for req in requirements)]
        return pool[rng.randrange(len(pool))] if pool else None

    def _descends(self, edge: JoinEdge) -> bool:
        left_col, right_col = edge.on[0]
        return (edge.left_table, left_col, edge.right_table, right_col) in self._descending

    def _chain(
        self,
        rng: random.Random,
        start: str,
        length: int,
        descending_only: bool = True,
    ) -> list[JoinEdge] | None:
        """A random FK chain of ``length`` edges leaving ``start``."""
        path: list[JoinEdge] = []
        current, used = start, {start}
        for _ in range(length):
            options: list[JoinEdge] = []
            for neighbour in sorted(self.join_graph.neighbours(current)):
                if neighbour in used or neighbour == self.code_table:
                    continue
                hop = self.join_graph.shortest_path(current, neighbour)
                if not hop or (descending_only and not self._descends(hop[0])):
                    continue
                options.append(hop[0])
            if not options:
                return None
            edge = options[rng.randrange(len(options))]
            path.append(edge)
            used.add(edge.right_table)
            current = edge.right_table
        return path

    def _group_axis(self, rng: random.Random, table: str) -> str | None:
        """A ``GROUP BY`` column on ``table``, biased toward coded dimensions."""
        roles = self._roles[table]
        # A named dimension (지점명) and a coded one (고객등급) both read well in
        # Korean; a free-form categorical is the weakest axis, so it is rarer.
        bag: list[str] = [*roles.labels * 3, *roles.codes * 3, *roles.categoricals]
        return bag[rng.randrange(len(bag))] if bag else None

    # ------------------------------------------------------------------ #
    # easy templates
    # ------------------------------------------------------------------ #

    @_template("easy_count_by_code", "easy")
    def _t_count_by_code(self, rng: random.Random) -> SQLProgram | None:
        table = self._source(rng)
        if table is None:
            return None
        column = self._pick(rng, self._roles[table].codes)
        if column is None:
            return None
        picked = self._code_value(rng, table, column)
        if picked is None:
            return None
        code, label = picked
        sql = f"SELECT COUNT(*) AS cnt\nFROM {table} a\nWHERE a.{column} = {_lit(code)}"
        return SQLProgram(
            sql=sql,
            difficulty="easy",
            tables=[table],
            template_id="easy_count_by_code",
            slots={
                "fact": table,
                "agg": "COUNT",
                "filter_col": f"{table}.{column}",
                "filter_value": code,
                "filter_label": label,
            },
        )

    @_template("easy_count_in_window", "easy")
    def _t_count_in_window(self, rng: random.Random) -> SQLProgram | None:
        table = self._source(rng)
        if table is None:
            return None
        column = self._pick(rng, self._roles[table].dates)
        if column is None:
            return None
        window = self._window(rng, table, column)
        if window is None:
            return None
        sql = (
            f"SELECT COUNT(*) AS cnt\nFROM {table} a\n"
            f"WHERE a.{column} BETWEEN {_lit(window.start)} AND {_lit(window.end)}"
        )
        return SQLProgram(
            sql=sql,
            difficulty="easy",
            tables=[table],
            template_id="easy_count_in_window",
            slots={"fact": table, "agg": "COUNT", **window.slots()},
        )

    @_template("easy_aggregate_in_window", "easy")
    def _t_aggregate_in_window(self, rng: random.Random) -> SQLProgram | None:
        table = self._source(rng, need_measure=True)
        if table is None:
            return None
        roles = self._roles[table]
        measure = self._pick(rng, roles.measures)
        column = self._pick(rng, roles.dates)
        if measure is None or column is None:
            return None
        window = self._window(rng, table, column)
        if window is None:
            return None
        agg = self._agg_for(rng, table, measure)
        sql = (
            f"SELECT {agg}(a.{measure}) AS val\nFROM {table} a\n"
            f"WHERE a.{column} BETWEEN {_lit(window.start)} AND {_lit(window.end)}"
        )
        return SQLProgram(
            sql=sql,
            difficulty="easy",
            tables=[table],
            template_id="easy_aggregate_in_window",
            slots={"fact": table, "agg": agg, "measure": f"{table}.{measure}", **window.slots()},
        )

    @_template("easy_aggregate_by_code", "easy")
    def _t_aggregate_by_code(self, rng: random.Random) -> SQLProgram | None:
        table = self._source(rng, need_measure=True)
        if table is None:
            return None
        roles = self._roles[table]
        measure = self._pick(rng, roles.measures)
        column = self._pick(rng, roles.codes)
        if measure is None or column is None:
            return None
        picked = self._code_value(rng, table, column)
        if picked is None:
            return None
        code, label = picked
        agg = self._agg_for(rng, table, measure)
        sql = f"SELECT {agg}(a.{measure}) AS val\nFROM {table} a\nWHERE a.{column} = {_lit(code)}"
        return SQLProgram(
            sql=sql,
            difficulty="easy",
            tables=[table],
            template_id="easy_aggregate_by_code",
            slots={
                "fact": table,
                "agg": agg,
                "measure": f"{table}.{measure}",
                "filter_col": f"{table}.{column}",
                "filter_value": code,
                "filter_label": label,
            },
        )

    @_template("easy_count_over_threshold", "easy")
    def _t_count_over_threshold(self, rng: random.Random) -> SQLProgram | None:
        table = self._source(rng, need_measure=True)
        if table is None:
            return None
        measure = self._pick(rng, self._roles[table].measures)
        if measure is None:
            return None
        threshold = self._threshold(rng, table, measure)
        if threshold is None:
            return None
        op = rng.choice((">=", ">", "<=", "<"))
        sql = f"SELECT COUNT(*) AS cnt\nFROM {table} a\nWHERE a.{measure} {op} {threshold}"
        return SQLProgram(
            sql=sql,
            difficulty="easy",
            tables=[table],
            template_id="easy_count_over_threshold",
            slots={
                "fact": table,
                "agg": "COUNT",
                "measure": f"{table}.{measure}",
                "op": op,
                "threshold": threshold,
            },
        )

    @_template("easy_distinct_count", "easy")
    def _t_distinct_count(self, rng: random.Random) -> SQLProgram | None:
        table = self._source(rng)
        if table is None:
            return None
        roles = self._roles[table]
        # A two-valued column (Y/N) makes COUNT(DISTINCT ...) a constant, so the
        # target needs real cardinality to be worth asking about.
        candidates = [
            c
            for c in (*roles.fk_columns, *roles.categoricals)
            if (self.profile.get(table, c) or ColumnProfile(table, c)).distinct_count >= 4
        ]
        target = self._pick(rng, candidates)
        column = self._pick(rng, roles.dates)
        if target is None or column is None:
            return None
        window = self._window(rng, table, column)
        if window is None:
            return None
        sql = (
            f"SELECT COUNT(DISTINCT a.{target}) AS cnt\nFROM {table} a\n"
            f"WHERE a.{column} BETWEEN {_lit(window.start)} AND {_lit(window.end)}"
        )
        return SQLProgram(
            sql=sql,
            difficulty="easy",
            tables=[table],
            template_id="easy_distinct_count",
            slots={
                "fact": table,
                "agg": "COUNT_DISTINCT",
                "distinct_col": f"{table}.{target}",
                **window.slots(),
            },
        )

    @_template("easy_top_n_rows", "easy")
    def _t_top_n_rows(self, rng: random.Random) -> SQLProgram | None:
        table = self._source(rng, need_measure=True)
        if table is None:
            return None
        roles = self._roles[table]
        measure = self._pick(rng, roles.measures)
        if measure is None or roles.key is None:
            return None
        column = self._pick(rng, roles.dates)
        window = self._window(rng, table, column) if column else None
        order = rng.choice(("DESC", "ASC"))
        limit = rng.choice((5, 10, 20))
        where = (
            f"WHERE a.{column} BETWEEN {_lit(window.start)} AND {_lit(window.end)}\n" if window else ""
        )
        sql = (
            f"SELECT a.{roles.key} AS id, a.{measure} AS val\nFROM {table} a\n{where}"
            f"ORDER BY a.{measure} {order}\nLIMIT {limit}"
        )
        slots: dict[str, Any] = {
            "fact": table,
            "measure": f"{table}.{measure}",
            "key_col": f"{table}.{roles.key}",
            "order": order,
            "limit": limit,
        }
        if window:
            slots.update(window.slots())
        return SQLProgram(sql, "easy", [table], "easy_top_n_rows", slots)

    # ------------------------------------------------------------------ #
    # medium templates
    # ------------------------------------------------------------------ #

    @_template("medium_group_by_dimension", "medium")
    def _t_group_by_dimension(self, rng: random.Random) -> SQLProgram | None:
        fact = self._source(rng, need_measure=True)
        if fact is None:
            return None
        chain = self._chain(rng, fact, 1)
        if chain is None:
            return None
        edge = chain[0]
        dim = edge.right_table
        axis = self._group_axis(rng, dim)
        measure = self._pick(rng, self._roles[fact].measures)
        if axis is None or measure is None:
            return None
        agg = "COUNT" if rng.random() < 0.3 else self._agg_for(rng, fact, measure)
        expression = "COUNT(*)" if agg == "COUNT" else f"{agg}(a.{measure})"
        date_col = self._pick(rng, self._roles[fact].dates)
        window = self._window(rng, fact, date_col) if date_col else None
        where = (
            f"WHERE a.{date_col} BETWEEN {_lit(window.start)} AND {_lit(window.end)}\n" if window else ""
        )
        limit = rng.choice((5, 10, 20))
        sql = (
            f"SELECT b.{axis} AS grp, {expression} AS val\n"
            f"FROM {fact} a\nJOIN {dim} b ON {edge.to_sql('a', 'b')}\n{where}"
            f"GROUP BY b.{axis}\nORDER BY val DESC\nLIMIT {limit}"
        )
        slots: dict[str, Any] = {
            "fact": fact,
            "dim": dim,
            "agg": agg,
            "measure": f"{fact}.{measure}",
            "group_col": f"{dim}.{axis}",
            "order": "DESC",
            "limit": limit,
        }
        if window:
            slots.update(window.slots())
        return SQLProgram(sql, "medium", [fact, dim], "medium_group_by_dimension", slots)

    @_template("medium_group_by_code_label", "medium")
    def _t_group_by_code_label(self, rng: random.Random) -> SQLProgram | None:
        if not self.code_table:
            return None
        fact = self._source(rng)
        if fact is None:
            return None
        column = self._pick(rng, self._roles[fact].codes)
        if column is None:
            return None
        edge = self.join_graph.code_join(fact, column, self.code_table)
        if edge is None:
            return None
        roles = self._roles[fact]
        measure = self._pick(rng, roles.measures)
        additive = measure is not None and self._is_additive(fact, measure)
        agg = "SUM" if additive and rng.random() < 0.5 else "COUNT"
        expression = "COUNT(*)" if agg == "COUNT" else f"SUM(a.{measure})"
        date_col = self._pick(rng, roles.dates)
        window = self._window(rng, fact, date_col) if date_col else None
        where = (
            f"WHERE a.{date_col} BETWEEN {_lit(window.start)} AND {_lit(window.end)}\n" if window else ""
        )
        sql = (
            f"SELECT x.{CODE_NAME_COLUMN} AS grp, {expression} AS val\n"
            f"FROM {fact} a\nJOIN {self.code_table} x ON {edge.to_sql('a', 'x')}\n{where}"
            f"GROUP BY x.{CODE_NAME_COLUMN}\nORDER BY val DESC"
        )
        slots: dict[str, Any] = {
            "fact": fact,
            "agg": agg,
            "group_col": f"{fact}.{column}",
            "code_table": self.code_table,
            "labelled": True,
        }
        if measure and agg == "SUM":
            slots["measure"] = f"{fact}.{measure}"
        if window:
            slots.update(window.slots())
        return SQLProgram(sql, "medium", [fact, self.code_table], "medium_group_by_code_label", slots)

    @_template("medium_date_bucket_trend", "medium")
    def _t_date_bucket_trend(self, rng: random.Random) -> SQLProgram | None:
        fact = self._source(rng)
        if fact is None:
            return None
        roles = self._roles[fact]
        date_col = self._pick(rng, roles.dates)
        if date_col is None:
            return None
        window = self._window(rng, fact, date_col)
        if window is None:
            return None
        bucket = "month" if window.kind in {"year", "half", "recent"} else "year"
        width = 6 if bucket == "month" else 4
        measure = self._pick(rng, roles.measures)
        additive = measure is not None and self._is_additive(fact, measure)
        extra = f", SUM(a.{measure}) AS val" if additive and rng.random() < 0.6 else ""
        sql = (
            f"SELECT substr(a.{date_col}, 1, {width}) AS bucket, COUNT(*) AS cnt{extra}\n"
            f"FROM {fact} a\n"
            f"WHERE a.{date_col} BETWEEN {_lit(window.start)} AND {_lit(window.end)}\n"
            f"GROUP BY bucket\nORDER BY bucket"
        )
        slots: dict[str, Any] = {"fact": fact, "agg": "COUNT", "bucket": bucket, **window.slots()}
        if extra:
            slots["measure"] = f"{fact}.{measure}"
        return SQLProgram(sql, "medium", [fact], "medium_date_bucket_trend", slots)

    @_template("medium_having_threshold", "medium")
    def _t_having_threshold(self, rng: random.Random) -> SQLProgram | None:
        fact = self._source(rng)
        if fact is None:
            return None
        chain = self._chain(rng, fact, 1)
        if chain is None:
            return None
        edge = chain[0]
        dim = edge.right_table
        axis = self._group_axis(rng, dim)
        if axis is None:
            return None
        date_col = self._pick(rng, self._roles[fact].dates)
        window = self._window(rng, fact, date_col) if date_col else None
        where = (
            f"WHERE a.{date_col} BETWEEN {_lit(window.start)} AND {_lit(window.end)}\n" if window else ""
        )
        having = rng.choice((10, 20, 50, 100))
        sql = (
            f"SELECT b.{axis} AS grp, COUNT(*) AS cnt\n"
            f"FROM {fact} a\nJOIN {dim} b ON {edge.to_sql('a', 'b')}\n{where}"
            f"GROUP BY b.{axis}\nHAVING COUNT(*) >= {having}\nORDER BY cnt DESC"
        )
        slots: dict[str, Any] = {
            "fact": fact,
            "dim": dim,
            "agg": "COUNT",
            "group_col": f"{dim}.{axis}",
            "having_op": ">=",
            "having_k": having,
        }
        if window:
            slots.update(window.slots())
        return SQLProgram(sql, "medium", [fact, dim], "medium_having_threshold", slots)

    @_template("medium_multi_predicate", "medium")
    def _t_multi_predicate(self, rng: random.Random) -> SQLProgram | None:
        fact = self._source(rng, need_measure=True)
        if fact is None:
            return None
        roles = self._roles[fact]
        code_col = self._pick(rng, roles.codes)
        date_col = self._pick(rng, roles.dates)
        measure = self._pick(rng, roles.measures)
        if code_col is None or date_col is None or measure is None:
            return None
        picked = self._code_value(rng, fact, code_col)
        window = self._window(rng, fact, date_col)
        threshold = self._threshold(rng, fact, measure)
        if picked is None or window is None or threshold is None:
            return None
        code, label = picked
        chain = self._chain(rng, fact, 1)
        join, tables, dim_slot = "", [fact], {}
        if chain is not None:
            edge = chain[0]
            join = f"JOIN {edge.right_table} b ON {edge.to_sql('a', 'b')}\n"
            tables.append(edge.right_table)
            dim_slot = {"dim": edge.right_table}
        op = rng.choice((">=", "<="))
        agg = "COUNT" if rng.random() < 0.4 else self._agg_for(rng, fact, measure)
        expression = "COUNT(*)" if agg == "COUNT" else f"{agg}(a.{measure})"
        sql = (
            f"SELECT {expression} AS val\nFROM {fact} a\n{join}"
            f"WHERE a.{code_col} = {_lit(code)}\n"
            f"  AND a.{date_col} BETWEEN {_lit(window.start)} AND {_lit(window.end)}\n"
            f"  AND a.{measure} {op} {threshold}"
        )
        slots: dict[str, Any] = {
            "fact": fact,
            "agg": agg,
            "measure": f"{fact}.{measure}",
            "filter_col": f"{fact}.{code_col}",
            "filter_value": code,
            "filter_label": label,
            "op": op,
            "threshold": threshold,
            **dim_slot,
            **window.slots(),
        }
        return SQLProgram(sql, "medium", tables, "medium_multi_predicate", slots)

    @_template("medium_ratio_nullif", "medium")
    def _t_ratio_nullif(self, rng: random.Random) -> SQLProgram | None:
        fact = self._source(rng)
        if fact is None:
            return None
        roles = self._roles[fact]
        code_col = self._pick(rng, roles.codes)
        axis = self._pick(rng, [c for c in roles.dimensions if c != code_col])
        if code_col is None or axis is None:
            return None
        picked = self._code_value(rng, fact, code_col)
        if picked is None:
            return None
        code, label = picked
        date_col = self._pick(rng, roles.dates)
        window = self._window(rng, fact, date_col) if date_col else None
        where = (
            f"WHERE a.{date_col} BETWEEN {_lit(window.start)} AND {_lit(window.end)}\n" if window else ""
        )
        sql = (
            f"SELECT a.{axis} AS grp,\n"
            f"       ROUND(CAST(SUM(CASE WHEN a.{code_col} = {_lit(code)} THEN 1 ELSE 0 END) AS REAL)\n"
            f"             / NULLIF(COUNT(*), 0), 4) AS ratio\n"
            f"FROM {fact} a\n{where}"
            f"GROUP BY a.{axis}\nORDER BY ratio DESC"
        )
        slots = {
            "fact": fact,
            "agg": "RATIO",
            "group_col": f"{fact}.{axis}",
            "filter_col": f"{fact}.{code_col}",
            "filter_value": code,
            "filter_label": label,
        }
        if window:
            slots.update(window.slots())
        return SQLProgram(sql, "medium", [fact], "medium_ratio_nullif", slots)

    @_template("medium_three_table_join", "medium")
    def _t_three_table_join(self, rng: random.Random) -> SQLProgram | None:
        fact = self._source(rng, need_measure=True)
        if fact is None:
            return None
        chain = self._chain(rng, fact, 2)
        if chain is None:
            return None
        first, second = chain
        far = second.right_table
        axis = self._group_axis(rng, far)
        measure = self._pick(rng, self._roles[fact].measures)
        if axis is None or measure is None:
            return None
        date_col = self._pick(rng, self._roles[fact].dates)
        window = self._window(rng, fact, date_col) if date_col else None
        where = (
            f"WHERE a.{date_col} BETWEEN {_lit(window.start)} AND {_lit(window.end)}\n" if window else ""
        )
        agg = "COUNT" if rng.random() < 0.3 else self._agg_for(rng, fact, measure)
        expression = "COUNT(*)" if agg == "COUNT" else f"{agg}(a.{measure})"
        limit = rng.choice((5, 10))
        sql = (
            f"SELECT c.{axis} AS grp, {expression} AS val\n"
            f"FROM {fact} a\n"
            f"JOIN {first.right_table} b ON {first.to_sql('a', 'b')}\n"
            f"JOIN {far} c ON {second.to_sql('b', 'c')}\n{where}"
            f"GROUP BY c.{axis}\nORDER BY val DESC\nLIMIT {limit}"
        )
        slots: dict[str, Any] = {
            "fact": fact,
            "dim": far,
            "bridge": first.right_table,
            "agg": agg,
            "measure": f"{fact}.{measure}",
            "group_col": f"{far}.{axis}",
            "limit": limit,
        }
        if window:
            slots.update(window.slots())
        return SQLProgram(sql, "medium", [fact, first.right_table, far], "medium_three_table_join", slots)

    # ------------------------------------------------------------------ #
    # hard templates
    # ------------------------------------------------------------------ #

    @_template("hard_above_global_average", "hard")
    def _t_above_global_average(self, rng: random.Random) -> SQLProgram | None:
        fact = self._source(rng, need_measure=True)
        if fact is None:
            return None
        roles = self._roles[fact]
        measure = self._pick(rng, roles.measures)
        date_col = self._pick(rng, roles.dates)
        if measure is None or date_col is None:
            return None
        window = self._window(rng, fact, date_col)
        if window is None:
            return None
        op = rng.choice((">", "<"))
        sql = (
            f"SELECT COUNT(*) AS cnt\nFROM {fact} a\n"
            f"WHERE a.{date_col} BETWEEN {_lit(window.start)} AND {_lit(window.end)}\n"
            f"  AND a.{measure} {op} (SELECT AVG({measure}) FROM {fact})"
        )
        return SQLProgram(
            sql,
            "hard",
            [fact],
            "hard_above_global_average",
            {"fact": fact, "agg": "COUNT", "measure": f"{fact}.{measure}", "op": op, **window.slots()},
        )

    @_template("hard_cte_two_stage", "hard")
    def _t_cte_two_stage(self, rng: random.Random) -> SQLProgram | None:
        fact = self._source(rng, need_measure=True)
        if fact is None:
            return None
        chain = self._chain(rng, fact, 1)
        if chain is None:
            return None
        edge = chain[0]
        dim = edge.right_table
        axis = self._group_axis(rng, dim)
        measure = self._pick(rng, self._roles[fact].measures)
        date_col = self._pick(rng, self._roles[fact].dates)
        if axis is None or measure is None or date_col is None:
            return None
        window = self._window(rng, fact, date_col)
        if window is None or not self._is_additive(fact, measure):
            return None
        limit = rng.choice((5, 10))
        sql = (
            "WITH per_group AS (\n"
            f"    SELECT b.{axis} AS grp, SUM(a.{measure}) AS amt\n"
            f"    FROM {fact} a\n    JOIN {dim} b ON {edge.to_sql('a', 'b')}\n"
            f"    WHERE a.{date_col} BETWEEN {_lit(window.start)} AND {_lit(window.end)}\n"
            f"    GROUP BY b.{axis}\n"
            ")\n"
            "SELECT grp, amt\nFROM per_group\n"
            "WHERE amt > (SELECT AVG(amt) FROM per_group)\n"
            f"ORDER BY amt DESC\nLIMIT {limit}"
        )
        return SQLProgram(
            sql,
            "hard",
            [fact, dim],
            "hard_cte_two_stage",
            {
                "fact": fact,
                "dim": dim,
                "agg": "SUM",
                "measure": f"{fact}.{measure}",
                "group_col": f"{dim}.{axis}",
                "limit": limit,
                **window.slots(),
            },
        )

    @_template("hard_period_over_period", "hard")
    def _t_period_over_period(self, rng: random.Random) -> SQLProgram | None:
        fact = self._source(rng, need_measure=True)
        if fact is None:
            return None
        chain = self._chain(rng, fact, 1)
        if chain is None:
            return None
        edge = chain[0]
        dim = edge.right_table
        axis = self._group_axis(rng, dim)
        measure = self._pick(rng, self._roles[fact].measures)
        date_col = self._pick(rng, self._roles[fact].dates)
        if axis is None or measure is None or date_col is None:
            return None
        current = self._window(rng, fact, date_col, offset_years=1)
        if current is None or current.kind in {"recent", "range"} or not self._is_additive(fact, measure):
            return None
        previous = _shift_window_years(current, -1)
        case_now = f"CASE WHEN a.{date_col} BETWEEN {_lit(current.start)} AND {_lit(current.end)}"
        case_prev = f"CASE WHEN a.{date_col} BETWEEN {_lit(previous.start)} AND {_lit(previous.end)}"
        sql = (
            f"SELECT b.{axis} AS grp,\n"
            f"       SUM({case_now} THEN a.{measure} ELSE 0 END) AS cur_amt,\n"
            f"       SUM({case_prev} THEN a.{measure} ELSE 0 END) AS prev_amt,\n"
            f"       ROUND(CAST(SUM({case_now} THEN a.{measure} ELSE 0 END) AS REAL)\n"
            f"             / NULLIF(SUM({case_prev} THEN a.{measure} ELSE 0 END), 0) - 1, 4) AS growth\n"
            f"FROM {fact} a\nJOIN {dim} b ON {edge.to_sql('a', 'b')}\n"
            f"WHERE a.{date_col} BETWEEN {_lit(previous.start)} AND {_lit(current.end)}\n"
            f"GROUP BY b.{axis}\nORDER BY growth DESC"
        )
        return SQLProgram(
            sql,
            "hard",
            [fact, dim],
            "hard_period_over_period",
            {
                "fact": fact,
                "dim": dim,
                "agg": "GROWTH",
                "measure": f"{fact}.{measure}",
                "group_col": f"{dim}.{axis}",
                **current.slots("date"),
                **previous.slots("prev"),
            },
        )

    @_template("hard_anti_join_not_exists", "hard")
    def _t_anti_join_not_exists(self, rng: random.Random) -> SQLProgram | None:
        # An anti-join needs the one→many direction: rows of the parent with no
        # matching child row.
        parent = self._source_with(
            rng, lambda t: bool(self._children(t)), lambda t: bool(self._roles[t].dates)
        )
        if parent is None:
            return None
        children = self._children(parent)
        edge = children[rng.randrange(len(children))]
        child = edge.right_table
        date_col = self._pick(rng, self._roles[parent].dates)
        if date_col is None:
            return None
        window = self._window(rng, parent, date_col)
        if window is None:
            return None
        conditions = " AND ".join(f"c.{rc} = a.{lc}" for lc, rc in edge.on)
        sql = (
            f"SELECT COUNT(*) AS cnt\nFROM {parent} a\n"
            f"WHERE a.{date_col} BETWEEN {_lit(window.start)} AND {_lit(window.end)}\n"
            f"  AND NOT EXISTS (\n      SELECT 1 FROM {child} c WHERE {conditions}\n  )"
        )
        return SQLProgram(
            sql,
            "hard",
            [parent, child],
            "hard_anti_join_not_exists",
            {"fact": parent, "child": child, "agg": "COUNT", **window.slots()},
        )

    @_template("hard_left_join_is_null", "hard")
    def _t_left_join_is_null(self, rng: random.Random) -> SQLProgram | None:
        parent = self._source_with(
            rng, lambda t: bool(self._children(t)), lambda t: bool(self._parents(t))
        )
        if parent is None:
            return None
        children, dims = self._children(parent), self._parents(parent)
        child_edge = children[rng.randrange(len(children))]
        dim_edge = dims[rng.randrange(len(dims))]
        if child_edge.right_table == dim_edge.right_table:
            return None
        child, dim = child_edge.right_table, dim_edge.right_table
        axis = self._group_axis(rng, dim)
        child_info = self.schema.table(child)
        fallback_key = child_info.primary_key[0] if child_info and child_info.primary_key else None
        child_key = self._roles[child].key or fallback_key
        date_col = self._pick(rng, self._roles[child].dates)
        if axis is None or child_key is None or date_col is None:
            return None
        window = self._window(rng, child, date_col)
        if window is None:
            return None
        on = " AND ".join(f"c.{rc} = a.{lc}" for lc, rc in child_edge.on)
        sql = (
            f"SELECT b.{axis} AS grp, COUNT(*) AS cnt\n"
            f"FROM {parent} a\nJOIN {dim} b ON {dim_edge.to_sql('a', 'b')}\n"
            f"LEFT JOIN {child} c ON {on}\n"
            f"     AND c.{date_col} BETWEEN {_lit(window.start)} AND {_lit(window.end)}\n"
            f"WHERE c.{child_key} IS NULL\n"
            f"GROUP BY b.{axis}\nORDER BY cnt DESC"
        )
        return SQLProgram(
            sql,
            "hard",
            [parent, dim, child],
            "hard_left_join_is_null",
            {
                "fact": parent,
                "dim": dim,
                "child": child,
                "agg": "COUNT",
                "group_col": f"{dim}.{axis}",
                **window.slots(),
            },
        )

    @_template("hard_nested_aggregate", "hard")
    def _t_nested_aggregate(self, rng: random.Random) -> SQLProgram | None:
        fact = self._source(rng, need_measure=True)
        if fact is None:
            return None
        chain = self._chain(rng, fact, 2)
        if chain is None:
            return None
        first, second = chain
        entity, dim = first.right_table, second.right_table
        entity_key = self._roles[entity].key
        axis = self._group_axis(rng, dim)
        measure = self._pick(rng, self._roles[fact].measures)
        date_col = self._pick(rng, self._roles[fact].dates)
        if entity_key is None or axis is None or measure is None or date_col is None:
            return None
        window = self._window(rng, fact, date_col)
        if window is None or not self._is_additive(fact, measure):
            return None
        outer = rng.choice(("MAX", "AVG"))
        limit = rng.choice((5, 10))
        sql = (
            f"SELECT s.grp, {outer}(s.amt) AS val\n"
            "FROM (\n"
            f"    SELECT c.{axis} AS grp, b.{entity_key} AS entity, SUM(a.{measure}) AS amt\n"
            f"    FROM {fact} a\n"
            f"    JOIN {entity} b ON {first.to_sql('a', 'b')}\n"
            f"    JOIN {dim} c ON {second.to_sql('b', 'c')}\n"
            f"    WHERE a.{date_col} BETWEEN {_lit(window.start)} AND {_lit(window.end)}\n"
            f"    GROUP BY c.{axis}, b.{entity_key}\n"
            ") s\n"
            f"GROUP BY s.grp\nORDER BY val DESC\nLIMIT {limit}"
        )
        return SQLProgram(
            sql,
            "hard",
            [fact, entity, dim],
            "hard_nested_aggregate",
            {
                "fact": fact,
                "entity": entity,
                "dim": dim,
                "agg": outer,
                "inner_agg": "SUM",
                "measure": f"{fact}.{measure}",
                "group_col": f"{dim}.{axis}",
                "entity_col": f"{entity}.{entity_key}",
                "limit": limit,
                **window.slots(),
            },
        )

    @_template("hard_rank_emulation", "hard")
    def _t_rank_emulation(self, rng: random.Random) -> SQLProgram | None:
        fact = self._source(rng, need_measure=True)
        if fact is None:
            return None
        chain = self._chain(rng, fact, 1)
        if chain is None:
            return None
        edge = chain[0]
        dim = edge.right_table
        axis = self._group_axis(rng, dim)
        measure = self._pick(rng, self._roles[fact].measures)
        date_col = self._pick(rng, self._roles[fact].dates)
        if axis is None or measure is None or date_col is None:
            return None
        window = self._window(rng, fact, date_col)
        if window is None:
            return None
        limit = rng.choice((5, 10))
        # The correlated count runs over the CTE (one row per dimension value),
        # not over the fact table, so rank emulation stays O(groups²).
        sql = (
            "WITH per_group AS (\n"
            f"    SELECT b.{axis} AS grp, SUM(a.{measure}) AS amt\n"
            f"    FROM {fact} a\n    JOIN {dim} b ON {edge.to_sql('a', 'b')}\n"
            f"    WHERE a.{date_col} BETWEEN {_lit(window.start)} AND {_lit(window.end)}\n"
            f"    GROUP BY b.{axis}\n"
            ")\n"
            "SELECT p.grp, p.amt,\n"
            "       (SELECT COUNT(*) FROM per_group q WHERE q.amt > p.amt) + 1 AS rnk\n"
            "FROM per_group p\n"
            f"ORDER BY rnk\nLIMIT {limit}"
        )
        return SQLProgram(
            sql,
            "hard",
            [fact, dim],
            "hard_rank_emulation",
            {
                "fact": fact,
                "dim": dim,
                "agg": "SUM",
                "measure": f"{fact}.{measure}",
                "group_col": f"{dim}.{axis}",
                "limit": limit,
                **window.slots(),
            },
        )

    @_template("hard_join_having_subquery", "hard")
    def _t_join_having_subquery(self, rng: random.Random) -> SQLProgram | None:
        fact = self._source(rng, need_measure=True)
        if fact is None:
            return None
        chain = self._chain(rng, fact, 2)
        if chain is None:
            return None
        first, second = chain
        entity, dim = first.right_table, second.right_table
        axis = self._group_axis(rng, dim)
        measure = self._pick(rng, self._roles[fact].measures)
        date_col = self._pick(rng, self._roles[fact].dates)
        if axis is None or measure is None or date_col is None:
            return None
        window = self._window(rng, fact, date_col)
        if window is None or not self._is_additive(fact, measure):
            return None

        # The IN-subquery filters the fact through a *third* relation that is not
        # part of the join tree — the shape a schema linker most often misses.
        sub = self._subquery_filter(rng, fact, exclude={entity, dim})
        if sub is None:
            return None
        sub_sql, sub_slots = sub
        having = rng.choice((5, 10, 20))
        sql = (
            f"SELECT c.{axis} AS grp, COUNT(*) AS cnt, SUM(a.{measure}) AS amt\n"
            f"FROM {fact} a\n"
            f"JOIN {entity} b ON {first.to_sql('a', 'b')}\n"
            f"JOIN {dim} c ON {second.to_sql('b', 'c')}\n"
            f"WHERE a.{date_col} BETWEEN {_lit(window.start)} AND {_lit(window.end)}\n"
            f"  AND {sub_sql}\n"
            f"GROUP BY c.{axis}\nHAVING COUNT(*) >= {having}\nORDER BY amt DESC"
        )
        return SQLProgram(
            sql,
            "hard",
            [fact, entity, dim, sub_slots["sub_table"]],
            "hard_join_having_subquery",
            {
                "fact": fact,
                "entity": entity,
                "dim": dim,
                "agg": "COUNT",
                "measure": f"{fact}.{measure}",
                "group_col": f"{dim}.{axis}",
                "having_op": ">=",
                "having_k": having,
                **sub_slots,
                **window.slots(),
            },
        )

    def _subquery_filter(
        self, rng: random.Random, fact: str, exclude: set[str] | None = None
    ) -> tuple[str, dict[str, Any]] | None:
        """``a.FK IN (SELECT PK FROM other WHERE <categorical predicate>)``.

        ``exclude`` keeps the sub-select off tables that are already joined in
        the outer query: rewriting ``b.CHNL_CD = '10'`` as a subquery against the
        same table is a tautology, not a harder example.
        """
        blocked = exclude or set()
        options: list[tuple[str, str, str, str]] = []
        for column in self._roles[fact].fk_columns:
            col = self.schema.column(fact, column)
            if col is None or col.foreign_key is None:
                continue
            parent, parent_key = col.foreign_key.to_table, col.foreign_key.to_column
            if parent == self.code_table:
                continue
            # Two shapes reach outside the join tree: filter through the parent
            # itself, or through a *sibling* fact that references the same parent
            # ("표준체로 인수된 계약의 청구 건수").  The sibling form is the one a
            # schema linker actually struggles with, so it is kept in the pool.
            reachable: list[tuple[str, str]] = [(parent, parent_key)]
            for other in sorted(self.schema.tables):
                info = self.schema.table(other)
                if info is None or other in {fact, parent} or other == self.code_table:
                    continue
                reachable.extend(
                    (other, fk.from_column)
                    for fk in info.foreign_keys
                    if fk.to_table == parent and fk.to_column == parent_key
                )
            for table, select_col in reachable:
                if table in blocked:
                    continue
                roles = self._roles[table]
                options.extend(
                    (column, table, select_col, candidate)
                    for candidate in (*roles.codes, *roles.categoricals)
                )
        if not options:
            return None
        fk_col, target, target_key, filter_col = options[rng.randrange(len(options))]
        picked = self._code_value(rng, target, filter_col)
        if picked is None:
            raw = self._categorical_value(rng, target, filter_col)
            if raw is None:
                return None
            picked = (raw, raw)
        value, label = picked
        sql = (
            f"a.{fk_col} IN (SELECT {target_key} FROM {target} WHERE {filter_col} = {_lit(value)})"
        )
        return sql, {
            "sub_table": target,
            "sub_col": f"{target}.{filter_col}",
            "sub_value": value,
            "sub_label": label,
        }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _as_entries(glossary: Any) -> list[GlossaryEntry]:
    """Accept a ``Glossary``, a bare list of entries, or ``None``."""
    if glossary is None:
        return []
    entries = getattr(glossary, "entries", glossary)
    return [e for e in entries if isinstance(e, GlossaryEntry)]


def _is_pii(col: ColumnInfo) -> bool:
    """Personal data never enters the corpus, in any clause."""
    if col.sensitivity in {Sensitivity.MASKED, Sensitivity.FORBIDDEN}:
        return True
    return bool(_PII_NAME_RE.search(col.name) or _PII_COMMENT_RE.search(col.comment or ""))


def _is_restricted(col: ColumnInfo) -> bool:
    """Aggregate-only columns: never a projection, never a ``GROUP BY`` axis."""
    if col.sensitivity is Sensitivity.INTERNAL:
        return True
    return bool(_RESTRICTED_COMMENT_RE.search(col.comment or ""))


def _is_number(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _round_nice(value: int) -> int:
    """Round to the unit a Korean analyst would actually say (만원 / 억원 단위)."""
    magnitude = abs(value)
    for floor_at, unit in ((100_000_000, 10_000_000), (10_000_000, 1_000_000), (1_000_000, 100_000),
                           (100_000, 10_000), (10_000, 1_000), (1_000, 100), (100, 10)):
        if magnitude >= floor_at:
            rounded = round(value / unit) * unit
            return int(rounded) or int(unit)
    return int(value)


def _lit(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _month_end(year: int, month: int) -> str:
    year, month = (year + 1, month - 12) if month > 12 else (year, month)
    last = date(year, 12, 31) if month == 12 else date(year, month + 1, 1) - timedelta(days=1)
    return last.strftime("%Y%m%d")


def _shift_months(yyyymmdd: str, delta: int) -> str:
    day = _to_date(yyyymmdd)
    total = day.year * 12 + (day.month - 1) + delta
    year, month = divmod(total, 12)
    month += 1
    last_day = int(_month_end(year, month)[6:])
    return date(year, month, min(day.day, last_day)).strftime("%Y%m%d")


def _shift_window_years(window: _Window, delta: int) -> _Window:
    return _Window(
        column=window.column,
        start=_shift_year(window.start, delta),
        end=_shift_year(window.end, delta),
        kind=window.kind,
        year=window.year + delta,
        part=window.part,
        n_months=window.n_months,
    )


def _shift_year(yyyymmdd: str, delta: int) -> str:
    year = int(yyyymmdd[:4]) + delta
    month, day = int(yyyymmdd[4:6]), int(yyyymmdd[6:8])
    last_day = int(_month_end(year, month)[6:])
    return date(year, month, min(day, last_day)).strftime("%Y%m%d")


def _to_date(yyyymmdd: str) -> date:
    return date(int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8]))


def _overlap_days(start: str, end: str, lo: str, hi: str) -> int:
    left, right = max(start, lo), min(end, hi)
    if left > right:
        return 0
    return (_to_date(right) - _to_date(left)).days + 1


def _allocate(n: int, mix: dict[str, float]) -> dict[str, int]:
    """Largest-remainder allocation so the counts sum to exactly ``n``."""
    weights = {d: max(0.0, float(mix.get(d, 0.0))) for d in DIFFICULTIES}
    total = sum(weights.values())
    if total <= 0:
        weights, total = dict(DEFAULT_MIX), sum(DEFAULT_MIX.values())
    exact = {d: n * w / total for d, w in weights.items()}
    counts = {d: int(v) for d, v in exact.items()}
    for difficulty in sorted(DIFFICULTIES, key=lambda d: exact[d] - counts[d], reverse=True):
        if sum(counts.values()) >= n:
            break
        counts[difficulty] += 1
    return counts

