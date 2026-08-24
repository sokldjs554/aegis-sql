"""Pre-generation ambiguity detection — deciding when *not* to answer.

A Text-to-SQL engine that always emits SQL is a liability in a regulated shop: an
under-specified question does not fail loudly, it returns a confident, plausible,
wrong number.  ``"실적"`` may mean 계약 건수 or 수입보험료; ``"월별 추이"`` with no
period silently scans the whole contract table; ``"그 고객"`` has no antecedent at
all.  Each of those is *unrecoverable downstream* — the SQL parses, executes and
returns rows, so verification, repair and execution-voting all pass.

So the check has to happen here, before a token is spent, and it has to be
explainable.  Every rule below contributes a fixed weight and one Korean reason
string; the weights are chosen so that a single *unanswerable* signal (missing
period on a trend question, an ambiguous metric, a dangling demonstrative) crosses
the threshold on its own, while merely *under-specified* signals (no ``LIMIT`` on a
ranking, an unscoped superlative) only do so in combination:

===========================  ======  ===========================================
rule                         weight  fires on its own?
===========================  ======  ===========================================
기간 (temporal, no range)      0.50   yes
지표 (ambiguous metric)        0.50   yes
지시 (dangling demonstrative)  0.50   yes
컬럼 (schema-link tie)         0.30   no
랭킹 (ranking without k)       0.25   no
범위 (unscoped superlative)    0.20   no
===========================  ======  ===========================================

Clarification text is built from templates, never from a model: the question the
user is asked must be reproducible and reviewable.  :meth:`AmbiguityDetector.
build_clarification` is exposed separately so a caller may swap in the
``clarify.user`` LLM prompt while keeping this as the deterministic fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from aegis_sql.nlu.korean import josa
from aegis_sql.observability.logging import get_logger
from aegis_sql.types import AmbiguityReport, LinkedSchema, NormalizedQuestion, SchemaGraph

log = get_logger("nlu.ambiguity")

#: Score at or above which the engine stops and asks.  Tunable; see the table above.
AMBIGUITY_THRESHOLD = 0.5

#: Score gap under which two competing column links are treated as a tie.
LINK_TIE_GAP = 0.05

_W_PERIOD = 0.50
_W_METRIC = 0.50
_W_DEIXIS = 0.50
_W_COLUMN = 0.30
_W_RANKING = 0.25
_W_SCOPE = 0.20

#: Domain metrics whose reading changes the answer.  ``question`` completes
#: "'<term>'은(는) ...", ``options`` becomes the clarification's choices.
_AMBIGUOUS_METRICS: dict[str, tuple[str, tuple[str, ...]]] = {
    "실적": ("계약 건수인지 보험료 금액인지", ("계약 건수", "월납보험료 합계", "총가입금액 합계")),
    "많은": ("건수가 많은 것인지 금액이 큰 것인지", ("건수 기준", "금액 기준")),
    "고객수": ("계약자 기준인지 피보험자 기준인지", ("계약자 기준", "피보험자 기준")),
    "매출": ("수입보험료인지 초회보험료인지", ("수입보험료", "초회보험료")),
}

#: When one of these appears the measure is already pinned, so ``"많은"`` is not vague.
_EXPLICIT_MEASURE = re.compile(r"보험료|금액|건수|개수|수납액|청구액|지급액|점수|일수|횟수")

_P_DEIXIS = re.compile(r"(그|저|해당|이|동)\s*(고객|계약|상품|건|지점|설계사|담보|청구|티켓|민원)")
#: Anything that could pin the referent down: an ID-shaped literal or a quoted value.
_P_IDENTIFIER = re.compile(r"[A-Za-z]{1,4}\d{3,}|\d{6,}|['\"“‘「『]")

_P_SUPERLATIVE = re.compile(r"가장|제일|최다|최고|최저|최대|최소")
_P_SCOPED = re.compile(r"중에서|가운데|내에서|중\s*(?:가장|최)|별\s*(?:최|1위)")

_P_TAG = re.compile(r"^\[(?P<tag>[가-힣]+)\]")
_P_QUOTED_KEY = re.compile(r"'([^']{1,60})'")

#: Highest-priority tag wins the single clarifying question we are allowed to ask.
_TAG_PRIORITY: tuple[str, ...] = ("지시", "지표", "기간", "컬럼", "랭킹", "범위")

_PERIOD_OPTIONS = ["최근 3개월", "올해 들어(연초~오늘)", "작년 한 해", "기간을 직접 지정"]
_RANKING_OPTIONS = ["상위 5건", "상위 10건", "전체 조회"]
_SCOPE_OPTIONS = ["전체 기준", "지점별 기준", "상품별 기준"]
_DEIXIS_OPTIONS = ["고객ID로 지정", "계약번호로 지정", "조건(기간·등급 등)으로 지정"]


@dataclass(slots=True)
class _Finding:
    tag: str
    weight: float
    reason: str


class AmbiguityDetector:
    """Rule-based clarification gate sitting between NLU and schema linking."""

    __slots__ = ("schema", "glossary_terms", "threshold")

    def __init__(
        self,
        schema: SchemaGraph,
        glossary_terms: list[str] | None = None,
        *,
        threshold: float = AMBIGUITY_THRESHOLD,
    ) -> None:
        self.schema = schema
        #: Terms the glossary already defines are, by construction, not ambiguous.
        self.glossary_terms = frozenset(t.strip() for t in (glossary_terms or []) if t.strip())
        self.threshold = threshold

    # -- public ------------------------------------------------------------ #

    def detect(self, nq: NormalizedQuestion, linked: LinkedSchema | None = None) -> AmbiguityReport:
        findings: list[_Finding] = []
        findings += self._rule_period(nq)
        findings += self._rule_metric(nq)
        findings += self._rule_deixis(nq)
        findings += self._rule_link_tie(linked)
        findings += self._rule_ranking(nq)
        findings += self._rule_scope(nq)

        score = min(1.0, sum(f.weight for f in findings))
        reasons = [f.reason for f in findings]
        report = AmbiguityReport(
            is_ambiguous=score >= self.threshold,
            reasons=reasons,
            score=round(score, 4),
        )
        if report.is_ambiguous:
            report.clarifying_question, report.options = self.build_clarification(reasons)
        log.debug("ambiguity checked", score=report.score, ambiguous=report.is_ambiguous, rules=len(reasons))
        return report

    def build_clarification(self, reasons: list[str]) -> tuple[str, list[str]]:
        """Turn reason strings into one Korean question plus 2-4 options, no LLM.

        Reasons are self-describing (``"[지표] '실적'은 ..."``), so this stays usable
        on a report that was persisted, replayed, or produced by another process.
        """
        by_tag: dict[str, str] = {}
        for reason in reasons:
            m = _P_TAG.match(reason)
            if m and m.group("tag") not in by_tag:
                by_tag[m.group("tag")] = reason
        if not by_tag:
            return "질문을 조금 더 구체적으로 알려주실 수 있을까요?", ["기간을 지정", "집계 기준을 지정"]

        tag = next((t for t in _TAG_PRIORITY if t in by_tag), next(iter(by_tag)))
        reason = by_tag[tag]
        keys = _P_QUOTED_KEY.findall(reason)

        if tag == "기간":
            question = "어느 기간을 기준으로 조회할까요?"
            options = list(_PERIOD_OPTIONS)
        elif tag == "지표":
            term = keys[0] if keys else "해당 지표"
            options = list(_AMBIGUOUS_METRICS.get(term, ("건수 기준", "금액 기준"))[1])
            question = f"'{term}'{josa(term, '을/를')} 어떤 기준으로 계산할까요?"
        elif tag == "지시":
            target = keys[0] if keys else "대상"
            question = f"'{target}'{josa(target, '이/가')} 어떤 대상인지 알려주세요."
            options = list(_DEIXIS_OPTIONS)
        elif tag == "컬럼":
            options = [self._describe_ref(ref) for ref in keys[:3]] or ["첫 번째 후보", "두 번째 후보"]
            question = "어느 항목을 기준으로 집계할까요?"
        elif tag == "랭킹":
            question = "몇 건까지, 무엇을 기준으로 정렬해서 보여드릴까요?"
            options = list(_RANKING_OPTIONS)
        else:  # 범위
            question = "어떤 범위에서 최상위를 찾을까요?"
            options = list(_SCOPE_OPTIONS)

        if len(by_tag) > 1:
            others = "·".join(t for t in _TAG_PRIORITY if t in by_tag and t != tag)
            if others:
                question = f"{question} (추가로 {others} 조건도 함께 알려주시면 한 번에 처리됩니다.)"
        return question, options[:4] if len(options) >= 2 else [*options, "직접 입력"]

    # -- rules -------------------------------------------------------------- #

    def _rule_period(self, nq: NormalizedQuestion) -> list[_Finding]:
        cues = _cues(nq)
        if not cues.get("temporal"):
            return []
        if nq.entities.get("date_range") or nq.entities.get("date_point"):
            return []
        return [
            _Finding(
                "기간",
                _W_PERIOD,
                "[기간] '추이/월별' 같은 시계열 표현이 있지만 조회 기간이 지정되지 않았습니다.",
            )
        ]

    def _rule_metric(self, nq: NormalizedQuestion) -> list[_Finding]:
        text = nq.normalized
        out: list[_Finding] = []
        for term, (why, _options) in _AMBIGUOUS_METRICS.items():
            if term not in text or term in self.glossary_terms:
                continue
            if term == "많은" and _EXPLICIT_MEASURE.search(text):
                continue
            out.append(
                _Finding(
                    "지표",
                    _W_METRIC,
                    f"[지표] '{term}'{josa(term, '은/는')} {why}에 따라 결과가 달라집니다.",
                )
            )
        return out[:1]  # one metric question at a time keeps the follow-up answerable

    def _rule_deixis(self, nq: NormalizedQuestion) -> list[_Finding]:
        m = _P_DEIXIS.search(nq.normalized)
        if not m or _P_IDENTIFIER.search(nq.normalized):
            return []
        phrase = f"{m.group(1)} {m.group(2)}"
        return [
            _Finding(
                "지시",
                _W_DEIXIS,
                f"[지시] '{phrase}'{josa(phrase, '이/가')} 무엇을 가리키는지 "
                "질문에 식별자(고객ID·계약번호)가 없습니다.",
            )
        ]

    def _rule_link_tie(self, linked: LinkedSchema | None) -> list[_Finding]:
        if linked is None:
            return []
        columns = sorted(
            (e for e in linked.evidence if "." in e.ref),
            key=lambda e: (-e.score, e.ref),
        )
        if len(columns) < 2:
            return []
        first, second = columns[0], columns[1]
        if abs(first.score - second.score) >= LINK_TIE_GAP:
            return []
        if first.ref.split(".")[0].upper() == second.ref.split(".")[0].upper():
            return []
        gap = abs(first.score - second.score)
        return [
            _Finding(
                "컬럼",
                _W_COLUMN,
                f"[컬럼] 후보 '{first.ref}', '{second.ref}'의 링킹 점수 차이가 "
                f"{gap:.3f}에 불과해 어느 테이블 기준인지 불분명합니다.",
            )
        ]

    def _rule_ranking(self, nq: NormalizedQuestion) -> list[_Finding]:
        cues = _cues(nq)
        if not cues.get("ranking") or nq.entities.get("top_k") is not None:
            return []
        if cues.get("aggregate") and _EXPLICIT_MEASURE.search(nq.normalized):
            return []
        return [
            _Finding(
                "랭킹",
                _W_RANKING,
                "[랭킹] 상위/하위를 물었지만 조회 건수와 정렬 기준이 명시되지 않았습니다.",
            )
        ]

    def _rule_scope(self, nq: NormalizedQuestion) -> list[_Finding]:
        text = nq.normalized
        if not _P_SUPERLATIVE.search(text):
            return []
        if nq.entities.get("group_by_hint") or _P_SCOPED.search(text):
            return []
        return [
            _Finding("범위", _W_SCOPE, "[범위] 최상급 표현의 비교 범위(전체/그룹 기준)가 없습니다.")
        ]

    # -- helpers ------------------------------------------------------------ #

    def _describe_ref(self, ref: str) -> str:
        """``TB_CTRT.MON_PRM`` → ``"계약의 월납보험료 (TB_CTRT.MON_PRM)"``."""
        if "." not in ref:
            return ref
        table_name, column_name = ref.split(".", 1)
        table = self.schema.table(table_name)
        column = self.schema.column(table_name, column_name) if table else None
        if table is None or column is None:
            return ref
        return f"{table.label}의 {column.label} ({ref})"


def _cues(nq: NormalizedQuestion) -> dict[str, bool]:
    cues = nq.entities.get("cues")
    return cues if isinstance(cues, dict) else {}
