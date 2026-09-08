"""Schema-capability answerability gate used by the EHRSQL-inspired research probe.

This module answers a narrower question than ambiguity detection:

    "The request is clear and safe, but does this database actually contain the
    information needed to answer it?"

The detector is intentionally deterministic and evidence-based. A capability
rule only fires when (1) the question names a concept and (2) the actual schema
+ business glossary do not expose evidence for that concept. The rules are a
small domain capability registry, not a claim that arbitrary out-of-domain
questions can be solved by keywords alone.
"""

from dataclasses import dataclass

_NORMALIZE_TABLE = str.maketrans("", "", " _-./()")


def _norm(text: str) -> str:
    return text.casefold().translate(_NORMALIZE_TABLE)


def _contains_group(text: str, group: tuple[str, ...]) -> bool:
    return all(_norm(term) in text for term in group)


@dataclass(frozen=True, slots=True)
class CapabilityRule:
    code: str
    query_groups: tuple[tuple[str, ...], ...]
    evidence_groups: tuple[tuple[str, ...], ...]
    reason: str


@dataclass(frozen=True, slots=True)
class AnswerabilityDecision:
    answerable: bool
    missing_concepts: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    matched_terms: tuple[str, ...] = ()


DEFAULT_CAPABILITY_RULES: tuple[CapabilityRule, ...] = (
    CapabilityRule("customer_credit_score", (("신용점수",), ("신용등급",)), (("신용점수",), ("신용등급",)), "고객 신용평가 정보가 현재 스키마/용어사전에 없습니다."),
    CapabilityRule("customer_occupation", (("직업군",), ("직군",), ("고객", "직업")), (("직업",), ("직군",)), "고객 직업·직군 정보가 현재 스키마/용어사전에 없습니다."),
    CapabilityRule("weather_event", (("태풍",), ("기상",), ("날씨",), ("강수",), ("폭우",)), (("기상",), ("날씨",), ("태풍",), ("강수",)), "외부 기상·재난 이벤트 데이터가 현재 스키마에 없습니다."),
    CapabilityRule("health_screening", (("건강검진",), ("검진결과",)), (("건강검진",), ("검진결과",)), "건강검진 결과 데이터가 현재 스키마에 없습니다."),
    CapabilityRule("competitor_product_price", (("경쟁사",), ("타사", "보험료"), ("타사", "상품")), (("경쟁사",), ("타사상품",), ("경쟁사보험료",)), "경쟁사 상품·가격 데이터가 현재 스키마에 없습니다."),
    CapabilityRule("customer_income", (("월소득",), ("연소득",), ("소득구간",), ("고객", "소득")), (("월소득",), ("연소득",), ("소득",)), "고객 소득 정보가 현재 스키마에 없습니다."),
    CapabilityRule("agent_training_hours", (("설계사", "교육"), ("교육", "이수시간")), (("교육이수",), ("교육시간",), ("교육이수시간",)), "설계사 교육 이수·교육시간 데이터가 현재 스키마에 없습니다."),
    CapabilityRule("card_spending", (("카드", "사용액"), ("카드", "소비"), ("신용카드", "사용액")), (("카드사용액",), ("카드소비액",), ("신용카드사용액",)), "외부 카드 소비·사용액 데이터가 현재 스키마에 없습니다."),
    CapabilityRule("housing_price", (("주택가격",), ("집값",), ("부동산", "가격")), (("주택가격",), ("집값",), ("부동산가격",)), "외부 주택·부동산 가격 데이터가 현재 스키마에 없습니다."),
    CapabilityRule("app_login_history", (("앱", "로그인"), ("로그인", "빈도"), ("접속", "빈도")), (("앱로그인",), ("로그인이력",), ("접속이력",)), "앱 로그인·디지털 행동 이력이 현재 스키마에 없습니다."),
    CapabilityRule("callcenter_agent_response_time", (("상담원", "응답시간"), ("콜센터", "응답시간")), (("상담원응답시간",), ("콜센터응답시간",)), "콜센터 상담원별 응답시간 데이터가 현재 스키마에 없습니다."),
    CapabilityRule("marketing_spend", (("광고비",), ("캠페인", "비용"), ("마케팅", "비용")), (("광고비",), ("캠페인비용",), ("마케팅비용",)), "광고비·캠페인 비용 데이터가 현재 스키마에 없습니다."),
    CapabilityRule("auto_accident_history", (("자동차", "사고", "이력"), ("교통사고", "이력")), (("자동차사고이력",), ("교통사고이력",)), "고객의 외부 자동차 사고 이력이 현재 스키마에 없습니다."),
    CapabilityRule("hospital_grade", (("의료기관", "등급"), ("병원", "등급")), (("의료기관등급",), ("병원등급",)), "병원명은 있어도 의료기관 등급 정보가 현재 스키마에 없습니다."),
    CapabilityRule("household_size", (("가족", "구성원"), ("가구원",), ("가구", "구성"), ("가족", "수")), (("가족구성원",), ("가구원수",), ("가구구성",)), "가구·가족 구성 정보가 현재 스키마에 없습니다."),
)


class SchemaAnswerabilityDetector:
    """Detect clear-but-unanswerable requests against the real domain schema."""

    __slots__ = ("rules", "_evidence_text")

    def __init__(
        self,
        schema,
        glossary_entries: list[object] | tuple[object, ...] | None = None,
        *,
        rules: tuple[CapabilityRule, ...] = DEFAULT_CAPABILITY_RULES,
    ) -> None:
        self.rules = rules
        evidence: list[str] = []
        for table in schema.tables.values():
            evidence.extend([table.name, table.comment or ""])
            for col in table.columns:
                evidence.extend([col.name, col.comment or ""])
        for entry in glossary_entries or ():
            evidence.extend(
                [
                    str(getattr(entry, "term", "")),
                    str(getattr(entry, "definition", "")),
                    *(str(v) for v in getattr(entry, "aliases", []) or []),
                    *(str(v) for v in getattr(entry, "columns", []) or []),
                    *(str(v) for v in getattr(entry, "tables", []) or []),
                ]
            )
        self._evidence_text = _norm(" ".join(evidence))

    def assess(self, question: str) -> AnswerabilityDecision:
        q = _norm(question)
        missing: list[str] = []
        reasons: list[str] = []
        matched: list[str] = []

        for rule in self.rules:
            query_match = next((g for g in rule.query_groups if _contains_group(q, g)), None)
            if query_match is None:
                continue
            evidence_present = any(_contains_group(self._evidence_text, g) for g in rule.evidence_groups)
            if evidence_present:
                continue
            missing.append(rule.code)
            reasons.append(rule.reason)
            matched.append("+".join(query_match))

        return AnswerabilityDecision(
            answerable=not missing,
            missing_concepts=tuple(missing),
            reasons=tuple(reasons),
            matched_terms=tuple(matched),
        )
