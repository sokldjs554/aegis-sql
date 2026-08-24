"""Deterministic slot-filling SQL compiler — the zero-cost tier of the cascade.

Why a rule-based generator exists in an LLM system
--------------------------------------------------
Three reasons, in order of how much they matter:

1. **It is the floor the whole repository stands on.**  CI has no API key, a
   reviewer's laptop has no GPU, and a production budget can be exhausted at
   11:00 on the last day of the month.  Every one of those situations still has
   to answer "작년 하반기에 체결된 계약 건수" correctly.  A generator that needs
   nothing but the schema, the profile and the 용어사전 keeps the engine alive in
   all three.
2. **It is the honest baseline.**  An ablation table whose first row is "no
   model at all" is the only way to show what the model tiers actually buy.  A
   toy baseline inflates every number above it; this one is built to be hard to
   beat on the head of the query distribution, where Korean insurance analytics
   is dominated by a handful of shapes: filter a fact table by a date window and
   a status code, aggregate one measure, group by one dimension, rank, cut.
3. **It is a semantic parser, not a fallback string.**  Everything it knows —
   that 실효 means ``CTRT_STAT_CD = '02'``, that 지점별 needs two FK hops through
   ``TB_AGNT``, that a ``CHAR(8)`` date column wants ``BETWEEN '20250701' AND
   '20251231'`` — is exactly the knowledge the prompt tiers must also be given.
   Encoding it as executable code rather than prose keeps it testable.

How resolution works
--------------------
The question arrives already analysed: :class:`~aegis_sql.types.NormalizedQuestion`
carries resolved date ranges, myriad-parsed amounts, comparison operators, group-by
dimensions and ``top_k``; :class:`~aegis_sql.types.LinkedSchema` carries the pruned
tables/columns with their scores and the glossary entries that fired.  This module
turns that into an intermediate representation and renders it:

1. **FROM** — the *grain* of the answer.  Scored as ``0.70·max(column evidence) +
   0.30·mean(top-3 column evidence) + 0.45·(FK degree ÷ max degree) + 0.25·(a
   confident glossary hit names the table)``.  The degree term is what stops
   "지점별 신계약 건수" from selecting ``FROM TB_BRCH`` — a dimension table with
   high lexical scores — while the evidence terms are what let "보험금 지급액
   합계" leave the hub table and select ``FROM TB_CLM``.
2. **Glossary predicates** — every confident hit with an ``sql_hint`` that parses
   as a *boolean* expression is injected verbatim, re-qualified with the alias of
   its owning table.  Hints that are value expressions (``SUM(MON_PRM) * 12``) are
   never used as filters, and hints with unbound ``:start`` / ``:end`` parameters
   are only used when the question supplied a date range to bind them to.
3. **Date predicates** — the date column is chosen from a Korean lexicon over the
   FROM table's ``CHAR(8)`` columns (체결→``CTRT_DT``, 청구→``CLM_DT``,
   접수→``RCPT_DT``, 수납→``PAY_DT``, 지급→``DEDT_DT``), rendered as ``BETWEEN``
   over ``'YYYYMMDD'`` string literals.  Several ranges on one column become an
   OR-group rather than a contradiction.
4. **Numeric predicates** — the measure column is resolved from the tokens
   immediately preceding the number (보험료→``MON_PRM``, 가입금액→``TOT_INSD_AMT``,
   청구금액→``CLM_AMT``, 지급금액→``PAY_AMT``, 위험점수→``RISK_SCR``,
   연체일수→``DLQ_DAYS``), falling back to Korean character-bigram similarity
   against the column's data-dictionary comment.
5. **Categorical predicates** — a value candidate that equals a code *label* in a
   profiled column becomes ``col = '<code>'``; one that equals a profiled value
   becomes an equality; anything else has to clear a schema-vocabulary filter
   before it is allowed a ``LIKE``.  A predicate invented from a word that was
   really a column name returns zero rows *silently*, which is the single worst
   failure mode a template generator can have — so the default is to emit nothing.
6. **Projection** — intent and cue driven: ``COUNT(*)``, ``SUM/AVG/MIN/MAX`` over
   the resolved measure, a ``CAST(SUM(CASE WHEN … END) AS REAL) / NULLIF(COUNT(*),0)``
   ratio, or, with no aggregate at all, the PK plus the five best-linked columns.
7. **GROUP BY** — 지점→``TB_BRCH.BRCH_NM`` (two hops), 상품→``TB_PROD.PROD_NM``,
   월/연도/분기→``substr`` over the chosen date column, 채널/등급/유형→the code
   column *plus* a ``LEFT JOIN TB_COMM_CD`` on ``(CD_GRP, CD)`` so the output
   reads ``대면`` and not ``10``.
8. **ORDER BY / LIMIT** — ranking cues pick the direction (상위·최다·많은 순 →
   ``DESC``; 하위·최소·적은 순 → ``ASC``), ``top_k`` picks the cut, and a
   month/quarter/year dimension is ordered chronologically instead.

Two structural decisions are worth calling out because they are where naive
slot-filling generators produce numbers that are wrong but plausible:

* **Linked ≠ joined.**  Schema linking is tuned for recall, so it hands over
  tables the question never mentions.  Only tables that carry a predicate, a
  dimension, a measure, or that *bridge* two such tables are joined; everything
  else is dropped.  A spurious join silently multiplies a ``COUNT``.
* **Fan-out becomes EXISTS.**  When a table contributes only filters and joins
  to the FROM grain on a non-unique key (a contract has many claims), the join
  is rewritten as a correlated ``EXISTS`` so that ``COUNT(*)`` still counts
  contracts.  Code-label joins are ``LEFT JOIN`` for the same reason: decoration
  must never change the cardinality.

The generator never raises.  If every resolution step fails it emits
``SELECT COUNT(*) FROM <best table>``, because a downstream repair loop can work
with a valid query and can do nothing at all with an exception.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, SqlglotError, TokenError

from aegis_sql.config import Settings
from aegis_sql.generation.base import GenerationContext
from aegis_sql.observability.logging import get_logger
from aegis_sql.schema.graph import JoinEdge, JoinGraph
from aegis_sql.schema.profile import SchemaProfile
from aegis_sql.types import (
    ColumnInfo,
    GenerationResult,
    GlossaryEntry,
    LinkedSchema,
    NormalizedQuestion,
    SchemaGraph,
    SQLCandidate,
    Tier,
    now_ms,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a retrieval import at runtime
    from aegis_sql.retrieval.glossary import Glossary

log = get_logger("generation.template_generator")

__all__ = ["TemplateGenerator", "QueryIR", "Predicate", "Projection", "PlannedJoin"]

PROMPT_VERSION = "template/v1"

#: Glossary hits below this score are lexical coincidences, not domain hits.
GLOSSARY_MIN_SCORE = 0.8
#: Default cut when a ranking cue appears without an explicit ``top_k``.
DEFAULT_TOP_K = 10
#: Marker substituted for the table qualifier while re-writing a glossary hint.
_ALIAS_MARKER = "AEGIS_ALIAS"

_PARSE_ERRORS = (ParseError, TokenError, SqlglotError, ValueError, RecursionError)
_NUMERIC_TYPES = frozenset({"INTEGER", "INT", "REAL", "NUMERIC", "DECIMAL", "FLOAT", "BIGINT", "DOUBLE"})
_WORD_RE = re.compile(r"[0-9a-zA-Z_]+|[가-힣]+")


# --------------------------------------------------------------------------- #
# Korean lexicons.  Longest surface first — matching is maximal-munch.
# --------------------------------------------------------------------------- #

#: Question surface → preferred physical measure columns, best first.
_MEASURE_LEXICON: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("월납보험료", ("MON_PRM",)),
    ("연납화보험료", ("MON_PRM",)),
    ("담보보험료", ("CVRG_PRM",)),
    ("기준보험료", ("BASE_PRM",)),
    ("총가입금액", ("TOT_INSD_AMT",)),
    ("담보가입금액", ("INSD_AMT",)),
    ("가입금액", ("TOT_INSD_AMT", "INSD_AMT")),
    ("보장금액", ("TOT_INSD_AMT", "INSD_AMT")),
    ("청구금액", ("CLM_AMT",)),
    ("지급금액", ("PAY_AMT",)),
    ("지급액", ("PAY_AMT",)),
    ("수납금액", ("PAY_AMT",)),
    ("보험금", ("PAY_AMT", "CLM_AMT")),
    ("보험료", ("MON_PRM", "CVRG_PRM", "BASE_PRM")),
    ("위험점수", ("RISK_SCR",)),
    ("이상징후점수", ("FRAUD_SCR",)),
    ("이상징후", ("FRAUD_SCR",)),
    ("만족도", ("SATIS_SCR",)),
    ("연체일수", ("DLQ_DAYS",)),
    ("납입기간", ("PAY_TERM_YR",)),
    ("부활횟수", ("RVIV_CNT",)),
    ("관리직원수", ("MNG_EMP_CNT",)),
    ("가입연령", ("MIN_AGE", "MAX_AGE")),
)

#: Question surface → preferred ``CHAR(8)`` date columns, best first.
_DATE_LEXICON: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("계약체결", ("CTRT_DT",)),
    ("판매개시", ("SALE_STRT_DT",)),
    ("판매종료", ("SALE_END_DT",)),
    ("담보개시", ("STRT_DT",)),
    ("담보종료", ("END_DT",)),
    ("납입기일", ("DUE_DT",)),
    ("처리완료", ("CMPL_DT",)),
    ("체결", ("CTRT_DT",)),
    ("가입", ("CTRT_DT", "JOIN_DT")),
    ("신계약", ("CTRT_DT",)),
    ("청구", ("CLM_DT",)),
    ("접수", ("CLM_DT", "RCPT_DT")),
    ("사고", ("ACDN_DT",)),
    ("지급", ("DEDT_DT", "PAY_DT")),
    ("수납", ("PAY_DT",)),
    ("납입", ("PAY_DT", "DUE_DT")),
    ("만기", ("EXPR_DT",)),
    ("해지", ("TERM_DT",)),
    ("실효", ("TERM_DT",)),
    ("심사", ("UW_DT",)),
    ("위촉", ("HIRE_DT",)),
    ("해촉", ("RSGN_DT",)),
    ("개설", ("OPEN_DT",)),
    ("폐쇄", ("CLS_DT",)),
    ("완료", ("CMPL_DT", "DEDT_DT")),
    ("민원", ("RCPT_DT",)),
    ("상담", ("RCPT_DT",)),
    ("계약", ("CTRT_DT",)),
)

#: ``별`` dimension → (table, column) preferences.  ``None`` table means "the
#: column lives on whichever required table has it".
_DIMENSION_LEXICON: dict[str, tuple[tuple[str | None, str], ...]] = {
    "지점": ((("TB_BRCH"), "BRCH_NM"),),
    "점포": ((("TB_BRCH"), "BRCH_NM"),),
    "지사": ((("TB_BRCH"), "BRCH_NM"),),
    "상품": ((("TB_PROD"), "PROD_NM"),),
    "설계사": ((("TB_AGNT"), "AGNT_NM"),),
    "모집인": ((("TB_AGNT"), "AGNT_NM"),),
    "담보": ((("TB_CVRG"), "CVRG_NM"),),
    "병원": ((("TB_CLM"), "HOSP_NM"),),
    "고객": ((None, "CUST_ID"),),
    "계약자": ((None, "CUST_ID"),),
    "채널": ((None, "CHNL_CD"),),
    "지역": ((None, "RGN_CD"),),
    "성별": ((None, "GNDR_CD"),),
    "등급": ((None, "VIP_GRD_CD"), (None, "GRD_CD")),
    "상태": ((None, "CTRT_STAT_CD"), (None, "CLM_STAT_CD"), (None, "PROC_STAT_CD")),
    "유형": ((None, "PROD_TYP_CD"), (None, "CLM_TYP_CD"), (None, "CTGY_CD")),
    "종류": ((None, "PROD_TYP_CD"), (None, "CLM_TYP_CD")),
    "납입방법": ((None, "PAY_MTHD_CD"),),
    "납입주기": ((None, "PAY_CYCL_CD"),),
    "결과": ((None, "UW_RSLT_CD"),),
    "진단": ((None, "DIAG_CD"),),
}

#: Dimensions that are a truncation of the date column rather than a column.
_DATE_DIMENSION: dict[str, tuple[int, str]] = {
    "일": (8, "일자"),
    "월": (6, "월"),
    "달": (6, "월"),
    "월별": (6, "월"),
    "연": (4, "연도"),
    "년": (4, "연도"),
    "연도": (4, "연도"),
    "년도": (4, "연도"),
}

#: Korean particles/endings glued onto a value candidate (실효 + 된).
_VALUE_SUFFIXES: tuple[str, ...] = (
    "에서", "으로", "된", "한", "인", "의", "이", "가", "은", "는", "을", "를",
    "와", "과", "도", "만", "에", "로",
)

_DESC_CUE_RE = re.compile(r"상위|최다|가장\s*많|제일\s*많|많은\s*순|높은\s*순|큰\s*순|내림차순|베스트|top\s*\d", re.I)
_ASC_CUE_RE = re.compile(r"하위|최소|가장\s*적|제일\s*적|적은\s*순|낮은\s*순|작은\s*순|오름차순")
_RANK_CUE_RE = re.compile(r"상위|하위|최다|최소|순위|랭킹|베스트|가장|제일|순으로|순서로|top\s*\d", re.I)
#: ``계약 수`` / ``고객수`` — a count request the intent classifier scores as "select".
_COUNT_TAIL_RE = re.compile(r"(?:계약|고객|건|개|명|사람|티켓|청구|담보)\s*수(?:는|가|를|요)?\s*[?？.]?\s*$")
_COUNT_CUE_RE = re.compile(r"건수|개수|몇\s*건|몇\s*개|몇\s*명|인원|카운트")
#: ``… 가장 큰 청구 10건`` — a trailing row count the NLU's ``top_k`` rules miss
#: because they require an explicit 상위/top marker.
_ROW_COUNT_RE = re.compile(r"(\d{1,4})\s*(?:건|개|명|위|가지)(?:\s*만)?\s*[.?!]?\s*$")
_DISTINCT_CUE_RE = re.compile(r"고유|중복\s*제거|서로\s*다른|distinct|유니크", re.I)
#: ``100건 이상`` after a group-by — a HAVING condition, not a row filter.
_COUNT_COMPARISON_RE = re.compile(r"(\d[\d,]*)\s*(?:건|개|명)\s*(이상|이하|초과|미만)")
_COUNT_CMP_OPS = {"이상": ">=", "이하": "<=", "초과": ">", "미만": "<"}
_QUOTED_RE = re.compile(r"[\"'“‘「『]([^\"'”’」』\n]{1,40})[\"'”’」』]")

_AGG_LABEL = {"COUNT": "건수", "SUM": "합계", "AVG": "평균", "MAX": "최대", "MIN": "최소"}


# --------------------------------------------------------------------------- #
# Intermediate representation
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Predicate:
    """A WHERE fragment carrying the alias slot it still needs filled.

    ``template`` holds ``AEGIS_ALIAS`` wherever the owning table's alias goes, so
    predicates can be resolved *before* the join plan assigns aliases — which is
    what lets a fan-out table be demoted to an ``EXISTS`` after the fact.
    """

    template: str
    table: str
    column: str = ""
    operator: str = ""
    origin: str = "glossary"  # glossary | date | numeric | categorical | exists

    def render(self, alias: str) -> str:
        return self.template.replace(_ALIAS_MARKER, alias)


@dataclass(slots=True)
class Projection:
    expression: str
    alias: str = ""
    is_aggregate: bool = False

    def render(self) -> str:
        return f'{self.expression} AS "{self.alias}"' if self.alias else self.expression


@dataclass(slots=True)
class PlannedJoin:
    edge: JoinEdge
    left_alias: str
    right_alias: str
    kind: str = "JOIN"  # "JOIN" | "LEFT JOIN"

    def render(self) -> str:
        on = self.edge.to_sql(self.left_alias, self.right_alias)
        return f"{self.kind} {self.edge.right_table} {self.right_alias} ON {on}"


@dataclass(slots=True)
class QueryIR:
    """The plan, one step before it becomes text."""

    from_: str
    select: list[Projection] = field(default_factory=list)
    joins: list[PlannedJoin] = field(default_factory=list)
    where: list[str] = field(default_factory=list)
    group_by: list[str] = field(default_factory=list)
    having: list[str] = field(default_factory=list)
    order_by: list[tuple[str, str]] = field(default_factory=list)
    limit: int | None = None
    #: Alias assigned to :attr:`from_`; kept for tracing/debugging.
    from_alias: str = "t1"

    def to_sql(self) -> str:
        parts = [
            "SELECT " + ", ".join(p.render() for p in self.select),
            f"FROM {self.from_} {self.from_alias}",
        ]
        parts.extend(j.render() for j in self.joins)
        if self.where:
            parts.append("WHERE " + " AND ".join(self.where))
        if self.group_by:
            parts.append("GROUP BY " + ", ".join(self.group_by))
        if self.having:
            parts.append("HAVING " + " AND ".join(self.having))
        if self.order_by:
            parts.append("ORDER BY " + ", ".join(f"{e} {d}" for e, d in self.order_by))
        if self.limit is not None:
            parts.append(f"LIMIT {self.limit}")
        return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #


class TemplateGenerator:
    """Grammar-based :class:`~aegis_sql.generation.base.Generator` — no model, no key."""

    tier: Tier = Tier.TEMPLATE
    name: str = "template"

    def __init__(
        self,
        schema: SchemaGraph,
        profile: SchemaProfile | None,
        join_graph: JoinGraph,
        glossary: "Glossary | Sequence[GlossaryEntry] | None",
        settings: Settings,
    ) -> None:
        self.schema = schema
        self.profile = profile
        self.join_graph = join_graph
        self.glossary_entries: list[GlossaryEntry] = _as_entries(glossary)
        self.settings = settings
        self._max_degree = max((join_graph.degree(t) for t in schema.tables), default=1) or 1
        self._vocabulary = self._build_vocabulary()

    # -- Generator protocol ------------------------------------------------ #

    def available(self) -> bool:
        """Always true: the tier exists precisely so that something always answers."""
        return True

    def generate(self, ctx: GenerationContext) -> GenerationResult:
        started = now_ms()
        nq = ctx.normalized or _minimal_question(ctx.question)
        linked = ctx.linked or LinkedSchema(tables=sorted(self.schema.tables))
        try:
            ir = self._plan(nq, linked, ctx)
            sql = self._render(ir, ctx.dialect or self.schema.dialect)
        except Exception as exc:  # pragma: no cover - the tier must never raise
            log.warning("template planning failed, using degenerate fallback", error=str(exc))
            sql = self._degenerate(linked)
            ir = None

        candidate = SQLCandidate(
            sql=sql,
            tier=Tier.TEMPLATE,
            raw_output=ir.to_sql() if ir else sql,
            prompt_version=PROMPT_VERSION,
        )
        log.info(
            "template generated",
            table=(ir.from_ if ir else "?"),
            joins=len(ir.joins) if ir else 0,
            predicates=len(ir.where) if ir else 0,
            grouped=bool(ir and ir.group_by),
        )
        # Deterministic by construction: sampling it n times would return n
        # identical strings and pollute self-consistency voting with fake agreement.
        return GenerationResult(
            candidates=[candidate],
            tier=Tier.TEMPLATE,
            model="template",
            latency_ms=now_ms() - started,
        )

    # ------------------------------------------------------------------ #
    # planning
    # ------------------------------------------------------------------ #

    def _plan(self, nq: NormalizedQuestion, linked: LinkedSchema, ctx: GenerationContext) -> QueryIR:
        text = nq.normalized or nq.raw
        tokens = nq.tokens or _WORD_RE.findall(text)
        entities: dict[str, Any] = nq.entities or {}
        scores = _evidence_scores(linked)

        available = self._available_tables(linked)
        from_table = self._choose_from(available, scores, linked, nq)
        columns = self._available_columns(linked, available, from_table)

        predicates: list[Predicate] = []
        claimed: set[tuple[str, str]] = set()  # (table, column) already filtered

        ratio_numerator = self._glossary_predicates(
            nq, linked, from_table, available, entities, predicates, claimed
        )
        self._date_predicates(nq, entities, from_table, predicates, claimed)
        having = self._having_conditions(text, entities)
        self._numeric_predicates(
            nq, entities, from_table, columns, predicates, claimed, skip=having.consumed
        )
        self._categorical_predicates(nq, from_table, columns, predicates, claimed)

        dimensions = self._dimensions(nq, entities, from_table, columns)
        measure = self._measure_column(nq, from_table, columns)
        aggregate = self._aggregate(nq, entities, text, dimensions, measure)

        # Everything that must be reachable from the FROM grain.
        required: list[str] = [from_table]
        required += [p.table for p in predicates]
        required += [d.table for d in dimensions if d.table]
        if measure is not None:
            required.append(measure.table)
        required = [t for t in dict.fromkeys(required) if t]

        plan = self._plan_joins(from_table, required, predicates, dimensions, measure)
        from_alias = plan.alias_of(from_table)
        assert from_alias is not None  # the FROM table is always claimed by _plan_joins
        ir = QueryIR(from_=from_table, from_alias=from_alias)

        ir.joins = plan.joins
        ir.where = plan.where
        ir.having = list(having.conditions)

        select, group_by, order_expr = self._projection(
            aggregate, measure, dimensions, plan, from_table, columns, ratio_numerator
        )
        ir.select = select
        ir.group_by = group_by

        self._ordering(nq, entities, text, ir, dimensions, order_expr, aggregate)
        return ir

    # -- step 1: the FROM grain ------------------------------------------- #

    def _available_tables(self, linked: LinkedSchema) -> list[str]:
        tables = [t for t in linked.tables if self.schema.table(t)]
        if not tables:
            tables = sorted(self.schema.tables)
        return tables

    def _available_columns(
        self, linked: LinkedSchema, available: Sequence[str], from_table: str
    ) -> list[ColumnInfo]:
        """Linked columns, widened to every column of the FROM table.

        Linking prunes aggressively and the FROM table is the one place where a
        missing column costs an entire query — the measure or the date column is
        almost always there even when its score fell below the cut.
        """
        out: list[ColumnInfo] = []
        seen: set[str] = set()
        for ref in linked.columns:
            table, _, column = ref.partition(".")
            col = self.schema.column(table, column) if column else None
            if col and col.qualified not in seen and col.table in available:
                seen.add(col.qualified)
                out.append(col)
        table_info = self.schema.table(from_table)
        for col in table_info.columns if table_info else []:
            if col.qualified not in seen:
                seen.add(col.qualified)
                out.append(col)
        return out

    def _choose_from(
        self,
        available: Sequence[str],
        scores: dict[str, float],
        linked: LinkedSchema,
        nq: NormalizedQuestion,
    ) -> str:
        gloss_tables = {
            t
            for entry in self._confident_glossary(nq, linked)
            for t in entry.tables
            if self.schema.table(t)
        }
        ranked: list[tuple[float, str]] = []
        for table in available:
            col_scores = sorted(
                (s for ref, s in scores.items() if ref.partition(".")[0] == table and "." in ref),
                reverse=True,
            )
            best = col_scores[0] if col_scores else 0.0
            top3 = sum(col_scores[:3]) / max(1, len(col_scores[:3])) if col_scores else 0.0
            degree = self.join_graph.degree(table) / self._max_degree
            score = 0.70 * best + 0.30 * top3 + 0.45 * degree
            score += 0.25 if table in gloss_tables else 0.0
            score += 0.10 * scores.get(table, 0.0)
            # ``-table`` in the key makes ties resolve alphabetically, not by
            # dict order, so the same question always compiles to the same SQL.
            ranked.append((score, table))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        log.debug("from-table ranking", ranking=[(t, round(s, 3)) for s, t in ranked[:4]])
        return ranked[0][1]

    # -- step 2: glossary --------------------------------------------------- #

    def _confident_glossary(self, nq: NormalizedQuestion, linked: LinkedSchema) -> list[GlossaryEntry]:
        """Glossary hits strong enough to change the SQL.

        ``LinkedSchema.glossary`` is recall-oriented and includes weak bag-of-token
        overlaps.  Injecting one of those as a ``WHERE`` clause changes the answer
        on the strength of a coincidence, so the evidence score gates it; when no
        evidence is attached (a hand-built ``LinkedSchema``), the term is required
        to actually occur in the question.
        """
        gloss_scores = {
            ref.split(":", 1)[1]: item_score
            for ref, item_score in _evidence_scores(linked).items()
            if ref.startswith("glossary:")
        }
        squished = re.sub(r"\s+", "", (nq.normalized or nq.raw)).lower()
        out: list[GlossaryEntry] = []
        for entry in linked.glossary:
            score = gloss_scores.get(entry.term)
            if score is None:
                surfaces = [entry.term, *entry.aliases]
                score = 1.0 if any(s.lower().replace(" ", "") in squished for s in surfaces) else 0.0
            if score >= GLOSSARY_MIN_SCORE:
                out.append(entry)
        return out

    def _glossary_predicates(
        self,
        nq: NormalizedQuestion,
        linked: LinkedSchema,
        from_table: str,
        available: Sequence[str],
        entities: dict[str, Any],
        predicates: list[Predicate],
        claimed: set[tuple[str, str]],
    ) -> Predicate | None:
        """Inject ``sql_hint`` fragments; return the one reserved for a ratio."""
        date_ranges = entities.get("date_range") or []
        bind = date_ranges[0][1] if date_ranges else None
        ratio_numerator: Predicate | None = None
        wants_ratio = nq.intent == "ratio" or bool((entities.get("cues") or {}).get("ratio"))

        for entry in self._confident_glossary(nq, linked):
            if not entry.sql_hint:
                continue
            owner = self._hint_owner(entry, from_table, available)
            if owner is None:
                continue
            built = self._compile_hint(entry.sql_hint, owner, bind)
            if built is None:
                continue
            template, column = built
            key = (owner, column)
            if key in claimed:
                continue
            claimed.add(key)
            predicate = Predicate(template, owner, column, origin="glossary")
            if wants_ratio and ratio_numerator is None:
                ratio_numerator = predicate  # becomes the CASE WHEN, not a filter
                continue
            predicates.append(predicate)
        return ratio_numerator

    def _hint_owner(self, entry: GlossaryEntry, from_table: str, available: Sequence[str]) -> str | None:
        candidates = [t for t in entry.tables if self.schema.table(t)]
        if from_table in candidates:
            return from_table
        for table in candidates:
            if table in available:
                return table
        return candidates[0] if candidates else None

    def _compile_hint(
        self, hint: str, owner: str, bind: tuple[str, str] | None
    ) -> tuple[str, str] | None:
        """Turn a glossary ``sql_hint`` into an alias-slotted boolean template.

        Returns ``None`` — meaning "do not use this hint" — for value expressions
        (``SUM(MON_PRM) * 12``), for hints whose columns are not on ``owner``, for
        hints that qualify columns with aliases this generator did not create, and
        for parameterised hints the question cannot bind.
        """
        body = hint.strip()
        if bind is not None:
            body = body.replace(":start", f"'{bind[0]}'").replace(":end", f"'{bind[1]}'")
        if ":" in body:
            return None
        try:
            tree = sqlglot.parse_one(body, read=self.schema.dialect)
        except _PARSE_ERRORS:
            return None
        if tree is None or not isinstance(tree, (exp.Predicate, exp.And, exp.Or, exp.Not, exp.Paren)):
            return None

        table = self.schema.table(owner)
        if table is None:
            return None
        first_column = ""
        for column in tree.find_all(exp.Column):
            if column.table:  # pre-qualified against an alias we do not control
                return None
            if table.column(column.name) is None:
                return None
            first_column = first_column or column.name
            column.set("table", exp.to_identifier(_ALIAS_MARKER))
        if not first_column:
            return None
        try:
            return tree.sql(dialect=self.schema.dialect), first_column
        except _PARSE_ERRORS:  # pragma: no cover - generator failure
            return None

    # -- step 3: dates ------------------------------------------------------ #

    def _date_predicates(
        self,
        nq: NormalizedQuestion,
        entities: dict[str, Any],
        from_table: str,
        predicates: list[Predicate],
        claimed: set[tuple[str, str]],
    ) -> None:
        ranges = [r for r in (entities.get("date_range") or []) if _is_range(r)]
        if not ranges:
            return
        column = self._date_column(nq, from_table)
        if column is None or (from_table, column.name) in claimed:
            return
        claimed.add((from_table, column.name))

        clauses = [
            f"{_ALIAS_MARKER}.{column.name} BETWEEN '{start}' AND '{end}'"
            for _surface, (start, end) in ranges
        ]
        # Several windows on one column are alternatives ("2024년과 2025년"),
        # never a conjunction — AND-ing them yields an empty result set.
        template = clauses[0] if len(clauses) == 1 else "(" + " OR ".join(clauses) + ")"
        predicates.append(Predicate(template, from_table, column.name, "BETWEEN", "date"))

    def _date_column(self, nq: NormalizedQuestion, table: str) -> ColumnInfo | None:
        candidates = [c for c in self._table_columns(table) if self._is_date_column(c)]
        if not candidates:
            return None
        text = (nq.normalized or nq.raw).replace(" ", "")
        by_name = {c.name.upper(): c for c in candidates}
        for surface, preferred in _DATE_LEXICON:
            if surface not in text:
                continue
            for name in preferred:
                if name in by_name:
                    return by_name[name]
        scored = sorted(
            candidates,
            key=lambda c: (-_label_similarity(text, c.label), c.name),
        )
        best = scored[0]
        if _label_similarity(text, best.label) > 0.0:
            return best
        # No signal at all: the table's own primary event date, in DDL order.
        not_null = [c for c in candidates if not c.nullable]
        return (not_null or candidates)[0]

    def _is_date_column(self, col: ColumnInfo) -> bool:
        cp = self.profile.get(col.table, col.name) if self.profile else None
        if cp is not None and cp.is_yyyymmdd:
            return True
        return col.name.upper().endswith("_DT") or col.name.upper() in {"BRDT"}

    # -- step 4: numeric ---------------------------------------------------- #

    def _numeric_predicates(
        self,
        nq: NormalizedQuestion,
        entities: dict[str, Any],
        from_table: str,
        columns: Sequence[ColumnInfo],
        predicates: list[Predicate],
        claimed: set[tuple[str, str]],
        skip: frozenset[int],
    ) -> None:
        comparisons = entities.get("comparison") or []
        if not comparisons:
            return
        for operator, value in comparisons:
            if isinstance(value, int) and value in skip:
                continue  # already expressed as a HAVING on the group count
            column = self._measure_column(nq, from_table, columns, near=value)
            if column is None:
                continue
            key = (column.table, column.name)
            if key in claimed:
                continue
            claimed.add(key)
            if operator == "between" and isinstance(value, (tuple, list)) and len(value) == 2:
                template = f"{_ALIAS_MARKER}.{column.name} BETWEEN {int(value[0])} AND {int(value[1])}"
            elif operator in {">=", "<=", ">", "<", "="} and isinstance(value, (int, float)):
                template = f"{_ALIAS_MARKER}.{column.name} {operator} {int(value)}"
            else:
                continue
            predicates.append(Predicate(template, column.table, column.name, operator, "numeric"))

    def _measure_column(
        self,
        nq: NormalizedQuestion,
        from_table: str,
        columns: Sequence[ColumnInfo],
        near: Any = None,
    ) -> ColumnInfo | None:
        """Resolve the numeric column a measure phrase refers to.

        ``near`` narrows the context window to the text immediately preceding the
        matched amount, which is where Korean puts the measure noun
        (``월납보험료가 20만원 이상``) — a whole-question scan would happily pick
        ``TOT_INSD_AMT`` out of an unrelated clause.
        """
        numeric = [c for c in columns if self._is_numeric(c) and not c.is_primary_key]
        if not numeric:
            return None
        context = self._context_window(nq, near)
        by_name: dict[str, list[ColumnInfo]] = {}
        for col in numeric:
            by_name.setdefault(col.name.upper(), []).append(col)

        for surface, preferred in _MEASURE_LEXICON:
            if surface not in context:
                continue
            for name in preferred:
                for col in by_name.get(name, []):
                    if col.table == from_table:
                        return col
                if by_name.get(name):
                    return by_name[name][0]

        scored = sorted(
            numeric,
            key=lambda c: (
                -_label_similarity(context, c.label),
                0 if c.table == from_table else 1,
                c.name,
            ),
        )
        best = scored[0]
        return best if _label_similarity(context, best.label) > 0.0 else None

    def _context_window(self, nq: NormalizedQuestion, near: Any) -> str:
        text = (nq.normalized or nq.raw).replace(" ", "")
        if near is None:
            return text
        surfaces = [s for s, value in (nq.entities.get("amount") or []) if value == near]
        for surface in surfaces:
            position = (nq.normalized or nq.raw).replace(" ", "").find(surface.replace(" ", ""))
            if position > 0:
                return text[max(0, position - 14) : position]
        return text

    def _is_numeric(self, col: ColumnInfo) -> bool:
        return col.dtype.upper() in _NUMERIC_TYPES

    # -- step 5: categorical ------------------------------------------------ #

    def _categorical_predicates(
        self,
        nq: NormalizedQuestion,
        from_table: str,
        columns: Sequence[ColumnInfo],
        predicates: list[Predicate],
        claimed: set[tuple[str, str]],
    ) -> None:
        if not self.profile:
            return
        quoted = {m.group(1).strip() for m in _QUOTED_RE.finditer(nq.raw)}
        for raw_value in list(quoted) + list(nq.value_candidates or []):
            value = _strip_value_suffix(raw_value)
            if len(value) < 2:
                continue
            hit = self._match_value(value, from_table, columns, claimed)
            if hit is None:
                if raw_value in quoted:
                    hit = self._match_name(value, from_table, columns, claimed, force=True)
                elif value.lower() not in self._vocabulary:
                    hit = self._match_name(value, from_table, columns, claimed, force=False)
            if hit is None:
                continue
            column, template, operator = hit
            claimed.add((column.table, column.name))
            predicates.append(Predicate(template, column.table, column.name, operator, "categorical"))

    def _match_value(
        self,
        value: str,
        from_table: str,
        columns: Sequence[ColumnInfo],
        claimed: set[tuple[str, str]],
    ) -> tuple[ColumnInfo, str, str] | None:
        """Exact hit against a code label or a profiled value — the strong path."""
        best: tuple[int, str, ColumnInfo, str] | None = None
        for col in columns:
            if (col.table, col.name) in claimed:
                continue
            cp = self.profile.get(col.table, col.name) if self.profile else None
            if cp is None:
                continue
            literal: str | None = None
            rank = 3
            for code, label in cp.code_labels.items():
                if label == value or code == value:
                    literal, rank = code, 0
                    break
            if literal is None and cp.is_categorical and value in cp.values:
                literal, rank = value, 1
            if literal is None:
                continue
            priority = rank * 2 + (0 if col.table == from_table else 1)
            if best is None or priority < best[0]:
                best = (priority, literal, col, "=")
        if best is None:
            return None
        _priority, literal, col, operator = best
        return col, f"{_ALIAS_MARKER}.{col.name} = '{_quote(literal)}'", operator

    def _match_name(
        self,
        value: str,
        from_table: str,
        columns: Sequence[ColumnInfo],
        claimed: set[tuple[str, str]],
        force: bool,
    ) -> tuple[ColumnInfo, str, str] | None:
        """``LIKE`` fallback, deliberately hard to trigger.

        A wrong ``LIKE`` returns an empty result set that looks like a legitimate
        answer, so a name predicate is only emitted when the question quoted the
        value, or when the value actually appears inside a profiled sample of a
        name column.  Everything else is treated as schema vocabulary.
        """
        name_columns = [
            c
            for c in columns
            if (c.table, c.name) not in claimed
            and c.dtype.upper() not in _NUMERIC_TYPES
            and (c.name.upper().endswith("_NM") or c.name.upper() in {"CUST_NM", "HOSP_NM"})
        ]
        if not name_columns:
            return None
        name_columns.sort(key=lambda c: (0 if c.table == from_table else 1, c.name))
        for col in name_columns:
            cp = self.profile.get(col.table, col.name) if self.profile else None
            sampled = any(value in sample for sample in (cp.values if cp else []))
            if force or sampled:
                return col, f"{_ALIAS_MARKER}.{col.name} LIKE '%{_quote(value)}%'", "LIKE"
        return None

    def _build_vocabulary(self) -> str:
        """Squished lower-case blob of every schema and glossary surface.

        Used as a negative filter: a token that already appears in a table name,
        a column comment or a business term is describing *the schema*, not a
        value stored in it, and must not become a ``LIKE`` predicate.
        """
        parts: list[str] = []
        for table in self.schema.tables.values():
            parts.extend([table.name, table.comment or ""])
            for col in table.columns:
                parts.extend([col.name, col.comment or ""])
        for entry in self.glossary_entries:
            parts.append(entry.term)
            parts.extend(entry.aliases)
        return re.sub(r"\s+", "", " ".join(parts)).lower()

    # -- step 6: aggregate -------------------------------------------------- #

    def _aggregate(
        self,
        nq: NormalizedQuestion,
        entities: dict[str, Any],
        text: str,
        dimensions: Sequence["_Dimension"],
        measure: ColumnInfo | None,
    ) -> str:
        """Pick the projection shape: ``RATIO | SUM | AVG | MAX | MIN | COUNT | TOPN | NONE``.

        ``TOPN`` is the one that is easy to get wrong.  "청구금액이 가장 큰 청구
        10건" and "최대 청구금액" share an intent label but not a shape: the first
        wants ten *rows* ordered by the measure, the second wants one scalar.  The
        row-count cue is what separates them.
        """
        cues = entities.get("cues") or {}
        intent = nq.intent
        if intent == "ratio" or cues.get("ratio"):
            return "RATIO"
        ranking = intent in {"rank", "max", "min"} or bool(_RANK_CUE_RE.search(text))
        if ranking and not dimensions and measure is not None and _top_k(entities, text) is not None:
            return "TOPN"
        if intent in {"sum", "avg", "max", "min"}:
            return intent.upper()
        if intent == "count" or _COUNT_CUE_RE.search(text) or _COUNT_TAIL_RE.search(text.strip()):
            return "COUNT"
        if intent == "rank" or entities.get("group_by_hint"):
            return "COUNT"
        return "NONE"

    # -- step 7: dimensions ------------------------------------------------- #

    def _dimensions(
        self,
        nq: NormalizedQuestion,
        entities: dict[str, Any],
        from_table: str,
        columns: Sequence[ColumnInfo],
    ) -> list["_Dimension"]:
        out: list[_Dimension] = []
        for hint in entities.get("group_by_hint") or []:
            dimension = self._resolve_dimension(hint, nq, from_table, columns)
            if dimension is not None and not any(d.key == dimension.key for d in out):
                out.append(dimension)
        return out

    def _resolve_dimension(
        self,
        hint: str,
        nq: NormalizedQuestion,
        from_table: str,
        columns: Sequence[ColumnInfo],
    ) -> "_Dimension | None":
        if hint in _DATE_DIMENSION:
            width, label = _DATE_DIMENSION[hint]
            column = self._date_column(nq, from_table)
            if column is None:
                return None
            expression = (
                f"{_ALIAS_MARKER}.{column.name}"
                if width == 8
                else f"substr({_ALIAS_MARKER}.{column.name}, 1, {width})"
            )
            return _Dimension(key=f"date:{column.name}:{width}", table=from_table,
                              expression=expression, label=label, temporal=True)
        if hint == "분기":
            column = self._date_column(nq, from_table)
            if column is None:
                return None
            ref = f"{_ALIAS_MARKER}.{column.name}"
            expression = (
                f"substr({ref}, 1, 4) || 'Q' || "
                f"CAST((CAST(substr({ref}, 5, 2) AS INTEGER) + 2) / 3 AS INTEGER)"
            )
            return _Dimension(key=f"quarter:{column.name}", table=from_table,
                              expression=expression, label="분기", temporal=True)

        for table_hint, column_name in _DIMENSION_LEXICON.get(hint, ()):
            column = self._find_column(column_name, table_hint, from_table, columns)
            if column is not None:
                return self._column_dimension(column, hint)

        # Nothing in the lexicon: fall back to the best-matching low-cardinality
        # column, which is what a "…별" question is nearly always asking for.
        scored = [
            (
                _label_similarity(hint, col.label),
                0 if col.table == from_table else 1,
                col,
            )
            for col in columns
            if self._is_groupable(col)
        ]
        scored = [item for item in scored if item[0] > 0.0]
        if not scored:
            return None
        scored.sort(key=lambda item: (-item[0], item[1], item[2].name))
        return self._column_dimension(scored[0][2], hint)

    def _column_dimension(self, column: ColumnInfo, hint: str) -> "_Dimension":
        label = column.label if column.label != column.name else hint
        if column.code_group and label.endswith("코드") and len(label) > 2:
            # The output column holds the *label*, so ``모집채널코드`` would be a lie.
            label = label[:-2]
        return _Dimension(
            key=column.qualified,
            table=column.table,
            expression=f"{_ALIAS_MARKER}.{column.name}",
            label=label,
            code_column=column if column.code_group else None,
        )

    def _find_column(
        self,
        column_name: str,
        table_hint: str | None,
        from_table: str,
        columns: Sequence[ColumnInfo],
    ) -> ColumnInfo | None:
        if table_hint:
            column = self.schema.column(table_hint, column_name)
            return column if column and self.join_graph.shortest_path(from_table, table_hint) is not None else None
        matches = [c for c in columns if c.name.upper() == column_name.upper()]
        matches.sort(key=lambda c: (0 if c.table == from_table else 1, c.table))
        return matches[0] if matches else None

    def _is_groupable(self, col: ColumnInfo) -> bool:
        if col.is_primary_key or self._is_numeric(col):
            return False
        cp = self.profile.get(col.table, col.name) if self.profile else None
        if cp is not None:
            return cp.is_categorical or bool(cp.code_labels)
        return col.name.upper().endswith(("_CD", "_NM", "_YN"))

    # -- join planning ------------------------------------------------------ #

    def _plan_joins(
        self,
        from_table: str,
        required: Sequence[str],
        predicates: Sequence[Predicate],
        dimensions: Sequence["_Dimension"],
        measure: ColumnInfo | None,
    ) -> "_JoinPlan":
        plan = _JoinPlan(from_table)
        edges = self._steiner_edges(from_table, required)

        # A table that only contributes filters and hangs off a non-unique key
        # would multiply the FROM grain; those become correlated EXISTS instead.
        projected = {d.table for d in dimensions if d.table}
        if measure is not None:
            projected.add(measure.table)
        exists_tables = {
            edge.right_table
            for edge in edges
            if edge.left_table == from_table
            and self._is_fanout(edge)
            and edge.right_table not in projected
            and not any(e.left_table == edge.right_table for e in edges)
        }

        for edge in edges:
            if edge.right_table in exists_tables:
                continue
            left_alias = plan.alias_of(edge.left_table)
            assert left_alias is not None  # edges are emitted in join order
            right_alias = plan.claim(edge.right_table)
            plan.joins.append(PlannedJoin(edge, left_alias, right_alias))

        for predicate in predicates:
            if predicate.table in exists_tables:
                continue
            alias = plan.alias_of(predicate.table)
            if alias is None:  # unreachable table — drop rather than cross join
                log.debug("dropping predicate on unjoinable table", table=predicate.table)
                continue
            plan.where.append(predicate.render(alias))

        for table in sorted(exists_tables):
            edge = next(e for e in edges if e.right_table == table and e.left_table == from_table)
            fragment = self._exists_fragment(edge, plan, predicates)
            if fragment:
                plan.where.append(fragment)

        # Dimensions on code columns get a LEFT JOIN to the code table so the
        # label is readable without ever changing the row count.
        for dimension in dimensions:
            if dimension.code_column is None:
                continue
            code_edge = self.join_graph.code_join(dimension.table, dimension.code_column.name)
            if code_edge is None:
                continue
            left_alias = plan.alias_of(dimension.table)
            if left_alias is None:
                continue
            right_alias = plan.claim_code(dimension.table, dimension.code_column.name)
            plan.joins.append(PlannedJoin(code_edge, left_alias, right_alias, kind="LEFT JOIN"))
            dimension.label_alias = right_alias
        return plan

    def _steiner_edges(self, from_table: str, required: Sequence[str]) -> list[JoinEdge]:
        """Steiner-tree edges from :meth:`JoinGraph.connect`, re-rooted at FROM.

        ``connect`` picks its own root (highest degree), which is the right choice
        for a bare table set but the wrong one here — the FROM table is already
        decided, and every edge has to be oriented outwards from it so aliases are
        assigned in join order.
        """
        wanted = [t for t in dict.fromkeys([from_table, *required]) if self.schema.table(t)]
        if len(wanted) <= 1:
            return []
        _ordered, edges = self.join_graph.connect(wanted)
        return _orient(edges, from_table)

    def _is_fanout(self, edge: JoinEdge) -> bool:
        """True when the joined side is not unique on its join key."""
        right = self.schema.table(edge.right_table)
        if right is None:
            return False
        keys = {k.upper() for k in right.primary_key}
        return not keys or not all(rc.upper() in keys for _lc, rc in edge.on)

    def _exists_fragment(
        self, edge: JoinEdge, plan: "_JoinPlan", predicates: Sequence[Predicate]
    ) -> str:
        alias = f"x{plan.next_exists()}"
        conditions = [
            f"{alias}.{rc} = {plan.alias_of(edge.left_table)}.{lc}" for lc, rc in edge.on
        ]
        conditions += [p.render(alias) for p in predicates if p.table == edge.right_table]
        return f"EXISTS (SELECT 1 FROM {edge.right_table} {alias} WHERE " + " AND ".join(conditions) + ")"

    # -- projection --------------------------------------------------------- #

    def _projection(
        self,
        aggregate: str,
        measure: ColumnInfo | None,
        dimensions: Sequence["_Dimension"],
        plan: "_JoinPlan",
        from_table: str,
        columns: Sequence[ColumnInfo],
        ratio_numerator: Predicate | None,
    ) -> tuple[list[Projection], list[str], str | None]:
        select: list[Projection] = []
        group_by: list[str] = []

        for dimension in dimensions:
            alias = plan.alias_of(dimension.table)
            if alias is None:
                continue
            expression = dimension.expression.replace(_ALIAS_MARKER, alias)
            if dimension.label_alias:
                label_expression = f"{dimension.label_alias}.CD_NM"
                select.append(Projection(label_expression, dimension.label))
                group_by.extend([expression, label_expression])
            else:
                select.append(Projection(expression, dimension.label))
                group_by.append(expression)

        order_expression: str | None = None
        if aggregate == "TOPN" and measure is not None:
            # Rank rows, not groups: the measure has to be visible in the output
            # for the ordering to be interpretable.
            select = self._fallback_projection(from_table, columns, plan, must_include=measure)
            alias = plan.alias_of(measure.table) or plan.from_alias
            return select, [], f"{alias}.{measure.name}"
        if aggregate == "RATIO":
            condition = (
                ratio_numerator.render(plan.alias_of(ratio_numerator.table) or plan.from_alias)
                if ratio_numerator and plan.alias_of(ratio_numerator.table)
                else None
            )
            expression = (
                f"CAST(SUM(CASE WHEN {condition} THEN 1 ELSE 0 END) AS REAL) / NULLIF(COUNT(*), 0)"
                if condition
                else "CAST(COUNT(*) AS REAL) / NULLIF(COUNT(*), 0)"
            )
            select.append(Projection(expression, "비율", is_aggregate=True))
            order_expression = expression
        elif aggregate == "COUNT":
            select.append(Projection("COUNT(*)", "건수", is_aggregate=True))
            order_expression = "COUNT(*)"
        elif aggregate in {"SUM", "AVG", "MAX", "MIN"} and measure is not None:
            alias = plan.alias_of(measure.table) or plan.from_alias
            expression = f"{aggregate}({alias}.{measure.name})"
            select.append(
                Projection(expression, f"{measure.label}_{_AGG_LABEL[aggregate]}", is_aggregate=True)
            )
            order_expression = expression
        elif aggregate in {"SUM", "AVG", "MAX", "MIN"}:
            # The question asked for a measure this schema does not expose here;
            # counting rows is the honest degradation.
            select.append(Projection("COUNT(*)", "건수", is_aggregate=True))
            order_expression = "COUNT(*)"

        if not select:
            select = self._fallback_projection(from_table, columns, plan)
        return select, group_by, order_expression

    def _fallback_projection(
        self,
        from_table: str,
        columns: Sequence[ColumnInfo],
        plan: "_JoinPlan",
        must_include: ColumnInfo | None = None,
    ) -> list[Projection]:
        """No aggregate was requested: show the key plus the best-linked columns."""
        alias = plan.from_alias
        table = self.schema.table(from_table)
        chosen: list[ColumnInfo] = []
        for name in table.primary_key if table else []:
            column = self.schema.column(from_table, name)
            if column is not None:
                chosen.append(column)
        if must_include is not None and must_include.table == from_table:
            chosen.append(must_include)
        picked = {c.qualified for c in chosen}
        for column in columns:
            if len(chosen) >= 6:
                break
            if column.table == from_table and column.qualified not in picked:
                picked.add(column.qualified)
                chosen.append(column)
        if not chosen:
            return [Projection("*")]
        return [Projection(f"{alias}.{c.name}", c.label) for c in chosen]

    # -- ordering ----------------------------------------------------------- #

    def _ordering(
        self,
        nq: NormalizedQuestion,
        entities: dict[str, Any],
        text: str,
        ir: QueryIR,
        dimensions: Sequence["_Dimension"],
        order_expression: str | None,
        aggregate: str,
    ) -> None:
        top_k = _top_k(entities, text)
        ranking = bool(_RANK_CUE_RE.search(text)) or top_k is not None or nq.intent == "rank"

        temporal = next((d for d in dimensions if d.temporal), None)
        if temporal is not None and not ranking and ir.group_by:
            ir.order_by = [(ir.group_by[0], "ASC")]  # a time series reads chronologically
        elif ranking and order_expression:
            direction = "ASC" if _ASC_CUE_RE.search(text) and not _DESC_CUE_RE.search(text) else "DESC"
            ir.order_by = [(order_expression, direction)]

        if top_k is not None:
            ir.limit = int(top_k)
        elif ranking and (ir.group_by or aggregate == "TOPN"):
            ir.limit = DEFAULT_TOP_K
        elif aggregate == "NONE":
            ir.limit = int(self.settings.verify.default_limit)

    # -- HAVING ------------------------------------------------------------- #

    def _having_conditions(self, text: str, entities: dict[str, Any]) -> "_Having":
        """``…이 100건 이상인 지점`` filters groups, not rows."""
        if not entities.get("group_by_hint"):
            return _Having([], frozenset())
        conditions: list[str] = []
        consumed: set[int] = set()
        for match in _COUNT_COMPARISON_RE.finditer(text):
            value = int(match.group(1).replace(",", ""))
            operator = _COUNT_CMP_OPS[match.group(2)]
            conditions.append(f"COUNT(*) {operator} {value}")
            consumed.add(value)
        return _Having(conditions, frozenset(consumed))

    # -- rendering ---------------------------------------------------------- #

    def _render(self, ir: QueryIR, dialect: str) -> str:
        raw = ir.to_sql()
        try:
            tree = sqlglot.parse_one(raw, read=dialect)
            if tree is not None:
                return tree.sql(dialect=dialect, pretty=True)
        except _PARSE_ERRORS as exc:
            log.warning("template produced unparsable SQL, degrading", error=str(exc), sql=raw[:200])
            return f"SELECT COUNT(*) AS \"건수\" FROM {ir.from_}"
        return raw

    def _degenerate(self, linked: LinkedSchema) -> str:
        table = next(
            (t for t in linked.tables if self.schema.table(t)),
            max(self.schema.tables, key=self.join_graph.degree) if self.schema.tables else "DUAL",
        )
        return f'SELECT COUNT(*) AS "건수" FROM {table}'

    # -- misc --------------------------------------------------------------- #

    def _table_columns(self, table: str) -> list[ColumnInfo]:
        info = self.schema.table(table)
        return list(info.columns) if info else []


# --------------------------------------------------------------------------- #
# planning helpers
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _Dimension:
    key: str
    table: str
    expression: str
    label: str
    code_column: ColumnInfo | None = None
    temporal: bool = False
    #: Alias of the ``TB_COMM_CD`` instance that carries this dimension's label.
    label_alias: str = ""


@dataclass(slots=True)
class _Having:
    conditions: list[str]
    consumed: frozenset[int]


class _JoinPlan:
    """Alias bookkeeping for one statement.

    Base tables take ``t1..tn`` in join order and code-table instances take
    ``cd1..cdn``; both counters are driven by traversal order, never by a set, so
    the emitted SQL is byte-identical across runs and processes.
    """

    __slots__ = ("from_table", "from_alias", "aliases", "code_aliases", "joins", "where", "_n", "_cd", "_x")

    def __init__(self, from_table: str) -> None:
        self.from_table = from_table
        self.aliases: dict[str, str] = {from_table: "t1"}
        self.code_aliases: dict[tuple[str, str], str] = {}
        self.joins: list[PlannedJoin] = []
        self.where: list[str] = []
        self.from_alias = "t1"
        self._n = 1
        self._cd = 0
        self._x = 0

    def claim(self, table: str) -> str:
        if table not in self.aliases:
            self._n += 1
            self.aliases[table] = f"t{self._n}"
        return self.aliases[table]

    def claim_code(self, table: str, column: str) -> str:
        key = (table, column)
        if key not in self.code_aliases:
            self._cd += 1
            self.code_aliases[key] = f"cd{self._cd}"
        return self.code_aliases[key]

    def alias_of(self, table: str) -> str | None:
        return self.aliases.get(table)

    def next_exists(self) -> int:
        self._x += 1
        return self._x


def _orient(edges: Sequence[JoinEdge], root: str) -> list[JoinEdge]:
    """BFS the Steiner edge set outwards from ``root``, flipping edges as needed."""
    adjacency: dict[str, list[tuple[str, JoinEdge]]] = {}
    for edge in edges:
        adjacency.setdefault(edge.left_table, []).append((edge.right_table, edge))
        adjacency.setdefault(edge.right_table, []).append((edge.left_table, _flip(edge)))

    ordered: list[JoinEdge] = []
    visited = {root}
    frontier = [root]
    while frontier:
        node = frontier.pop(0)
        for neighbour, edge in adjacency.get(node, []):
            if neighbour in visited:
                continue
            visited.add(neighbour)
            ordered.append(edge)
            frontier.append(neighbour)
    return ordered


def _flip(edge: JoinEdge) -> JoinEdge:
    return JoinEdge(
        left_table=edge.right_table,
        right_table=edge.left_table,
        on=[(right, left) for left, right in edge.on],
        literal_filters=[],
        kind=edge.kind,
    )


def _evidence_scores(linked: LinkedSchema) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in linked.evidence:
        out[item.ref] = max(out.get(item.ref, 0.0), float(item.score))
    return out


def _as_entries(glossary: Any) -> list[GlossaryEntry]:
    if glossary is None:
        return []
    entries = getattr(glossary, "entries", glossary)
    return [e for e in entries if isinstance(e, GlossaryEntry)]


def _minimal_question(question: str) -> NormalizedQuestion:
    """Stand-in when the pipeline calls the generator without the NLU stage."""
    return NormalizedQuestion(
        raw=question,
        normalized=question,
        tokens=_WORD_RE.findall(question),
        entities={},
        value_candidates=[],
        intent="select",
    )


def _top_k(entities: dict[str, Any], text: str) -> int | None:
    """The NLU's ``top_k``, widened by a trailing bare row count.

    ``상위 5개`` is caught upstream; ``… 가장 큰 청구 10건`` is not, because the
    number carries no 상위/top marker — only the ranking cue earlier in the
    sentence tells you it is a cut and not a quantity.
    """
    explicit = entities.get("top_k")
    if explicit is not None:
        return int(explicit)
    if not _RANK_CUE_RE.search(text):
        return None
    match = _ROW_COUNT_RE.search(text.strip())
    if match is None:
        return None
    value = int(match.group(1).replace(",", ""))
    return value if 0 < value <= 10000 else None


def _is_range(item: Any) -> bool:
    return (
        isinstance(item, (tuple, list))
        and len(item) == 2
        and isinstance(item[1], (tuple, list))
        and len(item[1]) == 2
    )


def _strip_value_suffix(value: str) -> str:
    """Drop one agglutinated Korean ending (``실효된`` → ``실효``)."""
    stripped = value.strip()
    for suffix in sorted(_VALUE_SUFFIXES, key=len, reverse=True):
        if stripped.endswith(suffix) and len(stripped) - len(suffix) >= 2:
            return stripped[: -len(suffix)]
    return stripped


def _quote(literal: str) -> str:
    return literal.replace("'", "''")


def _bigrams(text: str) -> set[str]:
    squished = re.sub(r"\s+", "", text)
    return {squished[i : i + 2] for i in range(len(squished) - 1)} or {squished}


def _label_similarity(context: str, label: str) -> float:
    """Containment-biased character-bigram overlap.

    Korean compounds concatenate (``월납보험료`` = 월납 + 보험료), so a bigram
    measure recovers the shared head that whole-token equality misses, and plain
    substring containment is treated as a certain match.
    """
    if not context or not label:
        return 0.0
    squished_context = re.sub(r"\s+", "", context)
    squished_label = re.sub(r"\s+", "", label)
    if squished_label and squished_label in squished_context:
        return 1.0
    a, b = _bigrams(squished_context), _bigrams(squished_label)
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))
