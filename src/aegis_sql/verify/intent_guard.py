"""Refusing the *request*, not just the generated statement.

The AST guard can only block what the generator actually emits.  That leaves a
hole: ask a read-only Text-to-SQL engine to *delete* a table and the generator —
which structurally cannot produce DML — quietly returns a harmless ``SELECT``
over the same table.  Nothing is blocked because nothing dangerous was written,
and the user gets an answer to a question they did not ask.

That is worse than a refusal.  A destructive request must be *named and
refused*, so this module classifies the intent of the natural-language request
before any SQL exists.  It sits in front of the pipeline; the AST guard still
runs behind it, and neither replaces the other:

    질문 의도 (여기)   →  "이 요청은 변경 요청입니다"        → 명시적 거부
    생성된 AST (guard) →  "이 SQL은 PII 컬럼을 참조합니다"   → 차단/마스킹

The hard part is precision, not recall.  Korean read requests are full of verbs
that look like writes — "정렬을 **바꿔서** 보여줘", "조건을 **수정해서** 다시
조회해줘" — so a naive keyword match would refuse ordinary questions.  The rule
here is deliberately asymmetric: unambiguous destructive verbs fire on their own,
while soft mutation verbs must be attached to a schema object *and* unopposed by
a read cue.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from aegis_sql.observability.logging import get_logger
from aegis_sql.types import NormalizedQuestion, SchemaGraph, Violation

log = get_logger("verify.intent")

IntentKind = Literal["read", "write", "admin", "pii"]

#: Surface forms of forbidden data that users actually type.  Derived terms are
#: added from the policy at construction time; these are the ones no schema
#: comment would give us.
_PII_SYNONYMS: dict[str, tuple[str, ...]] = {
    "주민등록번호": ("주민등록번호", "주민번호", "주민 번호", "실명번호", "고유식별정보", "rrn", "ssn"),
}

#: Verbs that have no read-only reading in a database request.
_HARD_WRITE = re.compile(
    r"(삭제(?!된|되었|됐)|지워|지우[세라고]|없애|날려|드롭|초기화|비워|폐기|"
    r"\bdelete\b|\bdrop\b|\btruncate\b|\binsert\s+into\b|\bupdate\s+\w+\s+set\b|\balter\s+table\b)",
    re.IGNORECASE,
)

#: Verbs that mutate *or* merely reshape a result — ambiguous on their own.
#: The negative lookahead drops adnominal/passive forms: "갱신**된** 계약",
#: "수정**된** 내역" describe existing data and are ordinary read requests.
#: "추가**로**" / "추가**적으로**" are adverbs, not a request to insert.
_SOFT_WRITE = re.compile(
    r"(수정|변경|바꿔|바꾸|업데이트|갱신|등록|입력|저장|반영"
    r"|추가(?!로|적|\s*로|\s*적)"
    r"|\bupdate\b|\binsert\b|\bset\b)"
    r"(?!된|되는|되어|되었|됐|됨|된다|하는|하던|했던|한\s)",
    re.IGNORECASE,
)

#: Presentation words that make a soft verb a *read* ("정렬을 바꿔서 보여줘").
_PRESENTATION = re.compile(
    r"(정렬|순서|기준|조건|형식|단위|포맷|컬럼\s*순|보기|화면|필터|sort|order|format)",
    re.IGNORECASE,
)

#: A mutation verb in the imperative, directly attached ("등록해줘", "반영해 주세요").
_IMPERATIVE = re.compile(
    r"(수정|변경|바꿔|바꾸|업데이트|갱신|추가|등록|입력|저장|반영)\s*(해\s*(줘|주세요|주시|라|다오)|하라|해라|할래)",
)

#: Read cues that indicate the user still wants data back.
_READ_CUE = re.compile(
    r"(보여|알려|조회|뽑아|출력|가져와|궁금|얼마|몇\s*건|몇\s*명|목록|리스트|현황|"
    r"통계|추이|비중|비율|\bselect\b|\bshow\b|\blist\b)",
    re.IGNORECASE,
)

#: Session/engine-level operations that are never in scope for a query engine.
_ADMIN = re.compile(
    r"(\bpragma\b|\battach\b|\bdetach\b|\bgrant\b|\brevoke\b|\bvacuum\b|\bload_extension\b|"
    r"권한\S*\s*(부여|회수|변경|주[어세])|계정\S*\s*(생성|삭제)|백업|복구|덤프)",
    re.IGNORECASE,
)


@dataclass(slots=True)
class RequestIntent:
    kind: IntentKind = "read"
    matched: list[str] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)
    score: float = 0.0

    @property
    def refused(self) -> bool:
        return self.kind != "read"


class RequestIntentGuard:
    """Classifies a natural-language request as read / write / admin."""

    #: A soft verb only counts as a mutation when it governs a schema object.
    SOFT_THRESHOLD = 0.6

    def __init__(self, schema: SchemaGraph, policy: Any = None, enabled: bool = True) -> None:
        self.schema = schema
        self.policy = policy
        self.enabled = enabled
        self._object_terms = self._collect_object_terms(schema)
        self._pii_terms = self._collect_pii_terms(schema, policy)

    def _collect_pii_terms(self, schema: SchemaGraph, policy: Any) -> dict[str, str]:
        """``{surface: qualified column}`` for every column the policy forbids.

        Silently answering "고객 이름이랑 주민등록번호 뽑아줘" with just the name
        is the same failure as answering a DELETE with a SELECT: the user asked
        for something and got a substitution they were never told about.  The
        AST guard cannot catch it — a generator that declines to emit the column
        produces a perfectly innocent statement — so the *request* has to be
        refused here.
        """
        terms: dict[str, str] = {}
        if policy is None:
            return terms
        for col in schema.all_columns:
            try:
                sensitivity = policy.sensitivity(col.table, col.name)
            except Exception:  # pragma: no cover - policy shape mismatch
                continue
            if getattr(sensitivity, "value", str(sensitivity)) != "forbidden":
                continue
            label = (col.comment or col.name).strip()
            surfaces = {label, label.replace("암호화", "").strip(), col.name.lower()}
            for canonical, synonyms in _PII_SYNONYMS.items():
                if canonical in label:
                    surfaces |= set(synonyms)
            for surface in surfaces:
                if len(surface) >= 3:
                    terms[surface.lower()] = col.qualified
        return terms

    @staticmethod
    def _collect_object_terms(schema: SchemaGraph) -> set[str]:
        """Physical names plus the Korean logical names a user would actually type."""
        terms: set[str] = set()
        for table in schema.tables.values():
            terms.add(table.name.lower())
            if table.comment:
                terms.add(table.comment)
            for col in table.columns:
                terms.add(col.name.lower())
                if col.comment:
                    terms.add(col.comment)
        terms |= {"테이블", "레코드", "데이터", "행", "컬럼", "필드", "값", "table", "row", "record"}
        return terms

    # -- classification ---------------------------------------------------- #

    def classify(self, nq: NormalizedQuestion) -> RequestIntent:
        text = nq.normalized or nq.raw
        intent = RequestIntent()
        if not self.enabled or not text.strip():
            return intent

        admin = _ADMIN.findall(text)
        if admin:
            intent.kind = "admin"
            intent.matched = [_flatten(m) for m in admin]
            intent.score = 1.0
            return intent

        lowered = text.lower()
        for surface, column in self._pii_terms.items():
            if surface in lowered:
                intent.kind = "pii"
                intent.matched = [surface]
                intent.objects = [column]
                intent.score = 1.0
                return intent

        hard = _HARD_WRITE.findall(text)
        if hard:
            intent.kind = "write"
            intent.matched = [_flatten(m) for m in hard]
            intent.objects = self._objects_in(text)
            intent.score = 1.0
            return intent

        soft = _SOFT_WRITE.findall(text)
        if not soft:
            return intent

        objects = self._objects_in(text)
        score = 0.5 if objects else 0.0
        # A read cue is a *veto*, not a missing bonus.  Refusing a legitimate
        # question costs more than missing a mutation the generator cannot
        # perform anyway — the AST guard is still behind this.
        if _READ_CUE.search(text):
            score -= 0.5
        if _PRESENTATION.search(text):
            score -= 0.4
        # "~로 바꿔줘" / "~으로 변경해줘": naming a target value is the giveaway
        # that the user means the stored value, not the presentation.
        if re.search(r"(으?로)\s*(수정|변경|바꿔|바꾸|갱신|업데이트|설정)", text):
            score += 0.4
        # An imperative attached directly to the verb ("등록해줘", "저장해 주세요").
        if _IMPERATIVE.search(text):
            score += 0.3

        if score >= self.SOFT_THRESHOLD:
            intent.kind = "write"
            intent.matched = [_flatten(m) for m in soft]
            intent.objects = objects
            intent.score = min(1.0, score)
        return intent

    def _objects_in(self, text: str) -> list[str]:
        lowered = text.lower()
        return sorted({t for t in self._object_terms if len(t) >= 2 and t in lowered})[:6]

    # -- pipeline entry point ---------------------------------------------- #

    def check(self, nq: NormalizedQuestion) -> Violation | None:
        intent = self.classify(nq)
        if not intent.refused:
            return None
        if intent.kind == "pii":
            column = intent.objects[0] if intent.objects else ""
            # 컬럼 참조는 ``Violation.subject`` 로 나가고 렌더러가 대괄호로 붙인다.
            # 메시지 본문에도 넣으면 CLI·콘솔 양쪽에서 같은 컬럼이 두 번 찍힌다.
            message = (
                f"요청하신 항목({intent.matched[0]})은 고유식별정보로 분류되어 조회할 수 없습니다. "
                "다른 항목으로 질문을 다시 작성해 주세요."
            )
            log.warning("request refused by intent guard", code="PII_REQUEST", matched=intent.matched)
            return Violation(
                code="PII_REQUEST", message=message, severity="block", subject=column or None
            )
        if intent.kind == "admin":
            message = (
                "데이터베이스 관리·권한·백업 작업은 조회 엔진의 범위를 벗어납니다. "
                "DBA 절차를 통해 요청하세요."
            )
            code = "ADMIN_INTENT"
        else:
            target = f" (대상: {', '.join(intent.objects)})" if intent.objects else ""
            message = (
                f"데이터 변경 요청으로 판단되어 거부했습니다{target}. "
                "이 엔진은 읽기 전용이며, 변경은 정식 변경관리 절차를 통해야 합니다."
            )
            code = "WRITE_INTENT"
        log.warning("request refused by intent guard", code=code, matched=intent.matched)
        return Violation(
            code=code,
            message=message,
            severity="block",
            subject=", ".join(intent.matched[:3]) or None,
        )


def _flatten(match: object) -> str:
    if isinstance(match, tuple):
        return next((m for m in match if m), "")
    return str(match)
