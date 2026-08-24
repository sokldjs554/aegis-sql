"""Difficulty classification and question decomposition (DIN-SQL, without the LLM call).

DIN-SQL (Pourreza & Rafiei, 2023) splits text-to-SQL into schema linking →
*classification & decomposition* → generation → self-correction, and its classifier
assigns one of three labels: ``EASY`` (single table), ``NON-NESTED COMPLEX`` (joins,
grouping) and ``NESTED COMPLEX`` (sub-queries / set operations), where only the last
class is broken into natural-language sub-questions before generation.

This module is that second stage, mapped one-to-one onto our vocabulary:

=========================  ========  =============================================
DIN-SQL class              here      what the generator receives
=========================  ========  =============================================
EASY                       ``EASY``  nothing extra
NON-NESTED COMPLEX         ``JOIN``  join path + grouping plan as sub-steps
NESTED COMPLEX             ``NESTED``  inner-aggregate → outer-filter sub-questions
=========================  ========  =============================================

The paper obtains the label with a prompted LLM.  We do it with rules, for two
reasons that matter more in production than the small accuracy difference: the label
sits on the critical path of *every* query (an extra round-trip costs latency on the
90% of questions that are EASY), and it is also a feature of the cascade router —
a non-deterministic label would make tier selection, cost budgeting and replay of a
logged trace all irreproducible.

The hard part is distinguishing a genuine sub-query from a literal predicate:
``"100만원보다 많은 계약"`` is a ``WHERE``, while ``"평균보다 많은 계약"`` is a
correlated sub-select.  The discriminator used here is whether the right-hand side of
the comparison resolved to a literal during normalisation
(:mod:`aegis_sql.nlu.korean` fills ``entities["comparison"]`` only when it did).
"""

from __future__ import annotations

import re

from aegis_sql.nlu.korean import josa
from aegis_sql.observability.logging import get_logger
from aegis_sql.schema.graph import JoinGraph
from aegis_sql.schema.introspect import CODE_TABLE_CANDIDATES
from aegis_sql.types import LinkedSchema, NormalizedQuestion, SchemaGraph

log = get_logger("nlu.decompose")

EASY = "EASY"
JOIN = "JOIN"
NESTED = "NESTED"

#: Comparison against an aggregate of the *whole* set — always a sub-query.
_P_AGG_COMPARE = re.compile(r"평균\s*보다|평균보다|전체\s*평균|전사\s*평균|평균\s*대비|전체\s*대비")
#: ``가장 많은 <A>의 <B>`` — argmax on A, then project B.  The genitive 의 is what
#: separates this from ``"가장 좋은 설계사는"``, which is a plain ORDER BY … LIMIT 1.
_P_ARGMAX_PROJECT = re.compile(
    r"가장\s*[가-힣]{1,4}\s*(?P<arg>[가-힣A-Za-z]{2,10})의\s*(?P<target>[가-힣A-Za-z]{2,12})"
)
_P_SCOPED_TOP = re.compile(r"(중에서|가운데|내에서)\s*(?:가장|최|상위|제일)")
_P_EXCLUDE = re.compile(r"제외하고|제외한|한\s*번도|없는\s*고객|아닌\s*[가-힣]{2,}")
_P_SHARE = re.compile(r"전체\s*대비|전사\s*대비|전체에서\s*차지|비중|점유")
_P_DIRECTION_UP = re.compile(r"높은|큰|많은|이상|초과|넘는")

#: Measure nouns, most specific first — the label used when writing sub-questions.
_MEASURE_TERMS: tuple[str, ...] = (
    "월납보험료", "총가입금액", "가입금액", "청구금액", "지급금액", "수납금액", "담보보험료",
    "기준보험료", "연납화보험료", "보험료", "이상징후점수", "만족도점수", "위험점수",
    "연체일수", "처리기간", "금액", "건수", "개수", "인원", "계약수",
)

_ENTITY_FALLBACK = "대상"


class QuestionDecomposer:
    """Rule-based DIN-SQL stage two.

    ``schema`` is optional so the classifier still works before introspection (the
    lexical table index simply stays empty and only linked evidence is used).
    """

    __slots__ = ("schema", "join_graph", "_table_terms")

    def __init__(self, schema: SchemaGraph | None = None, join_graph: JoinGraph | None = None) -> None:
        self.schema = schema
        self.join_graph = join_graph or (JoinGraph(schema) if schema else None)
        self._table_terms: dict[str, tuple[str, ...]] = _build_table_terms(schema)

    # -- classification ----------------------------------------------------- #

    def classify(self, nq: NormalizedQuestion, linked: LinkedSchema | None) -> str:
        if self._nested_kind(nq) is not None:
            return NESTED
        if self._join_reason(nq, linked) is not None:
            return JOIN
        return EASY

    def _nested_kind(self, nq: NormalizedQuestion) -> str | None:
        """Which nested shape this is, or ``None``.  Order encodes specificity."""
        text = nq.normalized
        cues = _cues(nq)
        if _P_AGG_COMPARE.search(text):
            return "agg_compare"
        if _P_ARGMAX_PROJECT.search(text):
            return "argmax_project"
        if _P_SCOPED_TOP.search(text):
            return "scoped_top"
        if _P_EXCLUDE.search(text) and cues.get("set_op"):
            return "exclude"
        if cues.get("ratio") and _P_SHARE.search(text):
            return "share"
        # A comparison cue whose right-hand side never resolved to a literal has to
        # be compared against something computed — i.e. a sub-select.
        if cues.get("comparison") and cues.get("nested") and not nq.entities.get("comparison"):
            return "implicit_agg"
        return None

    def _join_reason(self, nq: NormalizedQuestion, linked: LinkedSchema | None) -> str | None:
        tables = self._candidate_tables(nq, linked)
        if len(tables) >= 2:
            return f"{len(tables)}개 테이블 참조"
        for hint in nq.entities.get("group_by_hint", []):
            owners = self._tables_for_term(str(hint))
            if owners and not (owners & set(tables)):
                return f"그룹 기준 '{hint}'가 다른 테이블에 있음"
        return None

    # -- decomposition ------------------------------------------------------- #

    def decompose(self, nq: NormalizedQuestion, linked: LinkedSchema | None) -> list[str]:
        label = self.classify(nq, linked)
        if label == EASY:
            return []
        if label == NESTED:
            steps = self._decompose_nested(nq, self._nested_kind(nq) or "implicit_agg")
        else:
            steps = self._decompose_join(nq, linked)
        log.debug("question decomposed", label=label, steps=len(steps))
        return [f"{i}) {s}" for i, s in enumerate(steps, start=1)]

    def _decompose_nested(self, nq: NormalizedQuestion, kind: str) -> list[str]:
        text = nq.normalized
        metric = _metric(text)
        period = _period(nq)
        group = _group(nq) or self._entity_label(nq) or _ENTITY_FALLBACK
        entity = self._entity_label(nq) or _ENTITY_FALLBACK
        up = bool(_P_DIRECTION_UP.search(text))
        direction = "큰" if up else "작은"

        if kind == "agg_compare":
            return [
                f"{period} 전체의 {metric} 평균을 구한다.",
                f"{group}별 {metric}{josa(metric, '을/를')} 집계한다.",
                f"2)의 값이 1)의 평균보다 {direction} {group}만 남긴다.",
            ]
        if kind == "argmax_project":
            m = _P_ARGMAX_PROJECT.search(text)
            arg = m.group("arg") if m else entity
            target = m.group("target") if m else metric
            return [
                f"{period} {metric} 기준으로 가장 {direction} {arg}{josa(arg, '을/를')} 1건 찾는다.",
                f"1)에서 찾은 {arg}에 해당하는 {target}{josa(target, '을/를')} 조회한다.",
            ]
        if kind == "scoped_top":
            k = nq.entities.get("top_k") or 1
            return [
                f"{period} 조건에 맞는 {entity} 집합을 만든다.",
                f"1) 안에서 {metric} 기준 상위 {k}건을 고른다.",
            ]
        if kind == "exclude":
            return [
                f"제외 조건에 해당하는 {entity} 집합을 구한다.",
                f"{period} 전체 {entity}에서 1)의 집합을 뺀다.",
            ]
        if kind == "share":
            return [
                f"{period} 전체의 {metric} 합계를 구한다.",
                f"{group}별 {metric} 합계를 구한다.",
                "2)를 1)로 나누어 비중을 계산한다.",
            ]
        return [
            f"비교 기준이 되는 {metric} 집계값을 먼저 구한다.",
            f"각 {entity}의 {metric}{josa(metric, '을/를')} 1)의 값과 비교해 조건을 만족하는 행만 남긴다.",
        ]

    def _decompose_join(self, nq: NormalizedQuestion, linked: LinkedSchema | None) -> list[str]:
        ordered, joins = self._join_plan(self._candidate_tables(nq, linked))
        steps: list[str] = []
        labelled = ", ".join(self._table_label(t) for t in ordered) or "관련 테이블"
        if joins:
            steps.append(f"{labelled}{josa(labelled, '을/를')} {' AND '.join(joins)} 기준으로 조인한다.")
        else:
            steps.append(f"{labelled}에서 필요한 행을 선택한다.")

        filters = self._filters(nq)
        if filters:
            steps.append("조건을 적용한다: " + ", ".join(filters) + ".")

        metric = _metric(nq.normalized)
        group = _group(nq)
        if group:
            steps.append(f"{group}별로 {metric}{josa(metric, '을/를')} 집계한다.")
        elif _cues(nq).get("aggregate"):
            steps.append(f"{metric}{josa(metric, '을/를')} 집계한다.")

        top_k = nq.entities.get("top_k")
        if top_k is not None:
            steps.append(f"{metric} 내림차순으로 정렬해 상위 {top_k}건만 남긴다.")
        elif _cues(nq).get("ranking"):
            steps.append(f"{metric} 기준으로 정렬한다.")
        return steps

    def _filters(self, nq: NormalizedQuestion) -> list[str]:
        """Concrete predicates the generator must not drop, in reading order."""
        out: list[str] = []
        for surface, (start, end) in nq.entities.get("date_range") or []:
            out.append(f"{surface} → 날짜 BETWEEN '{start}' AND '{end}'")
        for op, value in nq.entities.get("comparison") or []:
            if op == "between" and isinstance(value, tuple):
                out.append(f"금액 BETWEEN {value[0]:,} AND {value[1]:,}")
            elif isinstance(value, int):
                out.append(f"금액 {op} {value:,}")
        for candidate in nq.value_candidates:
            # A token that *is* a schema label (계약, 고객, 상태) names a table or a
            # column; only tokens with no label of their own can be data literals.
            if self._tables_for_term(candidate):
                continue
            out.append(f"'{candidate}' 값 매칭")
            if len(out) >= 6:
                break
        return out

    # -- schema helpers ------------------------------------------------------ #

    def _candidate_tables(self, nq: NormalizedQuestion, linked: LinkedSchema | None) -> list[str]:
        """Linked tables when schema linking already ran, else lexical table hits.

        The measure noun counts as a table reference of its own: ``"지점별 월납보험료"``
        names one table by its label (지점 → ``TB_BRCH``) and another only through the
        column it aggregates (월납보험료 → ``TB_CTRT``), and missing the second one
        would misclassify a two-table question as EASY.
        """
        if linked and linked.tables:
            return [t for t in linked.tables if not _is_code_table(t)]
        tables = self._mentioned_tables(nq.normalized)
        metric = _metric_in_text(nq.normalized)
        if metric:
            owners = self._tables_for_term(metric)
            if owners and not owners & set(tables):
                tables.append(sorted(owners)[0])
        return tables

    def _mentioned_tables(self, text: str) -> list[str]:
        hits: list[str] = []
        for name, terms in self._table_terms.items():
            if any(term in text for term in terms):
                hits.append(name)
        return hits

    def _tables_for_term(self, term: str) -> set[str]:
        """Tables whose table- or column-level Korean label contains ``term``."""
        if not self.schema or not term:
            return set()
        owners: set[str] = set()
        for name, table in self.schema.tables.items():
            if _is_code_table(name):
                continue
            if term in (table.comment or ""):
                owners.add(name)
                continue
            if any(term in (col.comment or "") for col in table.columns):
                owners.add(name)
        return owners

    def _join_plan(self, tables: list[str]) -> tuple[list[str], list[str]]:
        """Steiner-connect the wanted tables; bridge tables belong in the plan too.

        ``지점별 월납보험료`` needs ``TB_BRCH`` and ``TB_CTRT``, but the FK path runs
        through ``TB_AGNT``.  Listing only the wanted tables while emitting a join
        condition on the bridge would hand the generator an inconsistent plan.
        """
        if not self.join_graph or len(tables) < 2:
            return tables, []
        ordered, edges = self.join_graph.connect(tables)
        conditions = [
            f"{edge.left_table}.{lc} = {edge.right_table}.{rc}"
            for edge in edges
            for lc, rc in edge.on
        ]
        return ordered or tables, conditions

    def _table_label(self, name: str) -> str:
        table = self.schema.table(name) if self.schema else None
        return f"{name}({table.comment})" if table and table.comment else name

    def _entity_label(self, nq: NormalizedQuestion) -> str | None:
        for name in self._mentioned_tables(nq.normalized):
            table = self.schema.table(name) if self.schema else None
            if table and table.comment:
                return table.comment.split("/")[0]
        return None


# --------------------------------------------------------------------------- #
# Module-private helpers
# --------------------------------------------------------------------------- #


def _build_table_terms(schema: SchemaGraph | None) -> dict[str, tuple[str, ...]]:
    """Korean surface forms that imply each table, taken from the data dictionary."""
    if not schema:
        return {}
    out: dict[str, tuple[str, ...]] = {}
    for name, table in schema.tables.items():
        if _is_code_table(name):
            continue
        comment = table.comment or ""
        terms = tuple(t for t in re.split(r"[/·,]", comment) if len(t) >= 2)
        if terms:
            out[name] = terms
    return out


def _is_code_table(name: str) -> bool:
    return name.upper() in {c.upper() for c in CODE_TABLE_CANDIDATES}


def _cues(nq: NormalizedQuestion) -> dict[str, bool]:
    cues = nq.entities.get("cues")
    return cues if isinstance(cues, dict) else {}


def _metric_in_text(text: str) -> str | None:
    """The most specific measure noun literally present, or ``None``."""
    return next((term for term in _MEASURE_TERMS if term in text), None)


def _metric(text: str) -> str:
    return _metric_in_text(text) or "건수"


def _group(nq: NormalizedQuestion) -> str | None:
    hints = nq.entities.get("group_by_hint") or []
    return str(hints[0]) if hints else None


def _period(nq: NormalizedQuestion) -> str:
    ranges = nq.entities.get("date_range") or []
    return str(ranges[0][0]) if ranges else "지정된 기간"


