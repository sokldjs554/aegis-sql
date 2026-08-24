"""SQL → natural Korean back-translation, deterministically and without an API key.

The sampler emits programs; a training corpus needs the question a person would
have asked to get them.  Doing that with an LLM is the obvious move and the wrong
default, for three reasons that matter more than fluency:

1. **The pipeline must run with no credentials.**  CI, a reviewer's laptop and a
   cost-frozen month all still have to be able to rebuild the dataset.  A
   flywheel that stops turning without a billing account is not a flywheel.
2. **Back-translation is where silent label noise enters.**  A model that renders
   ``BETWEEN '20250701' AND '20251231'`` as "작년에" produces a pair whose
   question and SQL disagree — and execution-based verification cannot catch it,
   because the SQL still runs and still returns rows.  Rendering the window from
   the slot bindings makes that class of error unreachable.
3. **Determinism is a dataset property.**  Re-running the build must reproduce
   the corpus byte-for-byte, or an ablation between two model checkpoints is
   confounded by a silently different training set.

So the deterministic realiser is the *primary* path and the LLM is an optional
rewriter layered on top: when a client is supplied, ``backtranslate.user`` is
asked to make the sentence sound more like a working analyst, the result is
validated (Korean, single sentence, no physical identifiers leaking through), and
the deterministic string is kept whenever validation fails.

Where the Korean comes from
---------------------------
Never from the physical names.  The realiser reads the data dictionary
(``CTRT_STAT_CD`` → 계약상태코드) and the 사내 용어사전, and it prefers the
glossary: a predicate ``CTRT_STAT_CD = '02'`` is matched against the entries'
``sql_hint`` fragments and comes out as 실효 (or one of its aliases 효력상실 /
실효계약), not as "계약상태코드가 02인".  That is the vocabulary the questions at
inference time will use, so it is the vocabulary the corpus has to be written in.

Each ``template_id`` has several surface realisations selected by an RNG seeded
from ``(seed, template_id, sql)``, which keeps the corpus varied while keeping
each program's question stable across runs and across process boundaries — a
property ``hash()`` would not give.

Relative time expressions ("최근 6개월간") are anchored to the *extract's* last
observed date rather than to the wall clock, which is the same anchor
``nlu.korean`` uses at inference time (``today``).  Nothing here reads the clock.
"""

from __future__ import annotations

import random
import re
from typing import Any

from aegis_sql.flywheel.augment import josa, particle
from aegis_sql.flywheel.sql_sampler import SQLProgram
from aegis_sql.llm.base import LLMClient, Message
from aegis_sql.observability.logging import get_logger
from aegis_sql.prompts.registry import PromptRegistry
from aegis_sql.schema.card import SchemaCardBuilder
from aegis_sql.schema.profile import SchemaProfile
from aegis_sql.types import GlossaryEntry, LinkedSchema, SchemaGraph

log = get_logger("flywheel.back_translate")

__all__ = ["BackTranslator", "BACKTRANSLATE_PROMPT"]

#: Prompt id used when an LLM rewriter is supplied.
BACKTRANSLATE_PROMPT = "backtranslate.user"

_HANGUL_RE = re.compile(r"[가-힣]")
_WS_RE = re.compile(r"\s+")
_IDENT_RE = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")

#: Suffixes stripped when a column label is used as a dimension name
#: (계약상태코드 → 계약상태, 지점명 → 지점).
_LABEL_SUFFIX_RE = re.compile(r"(코드|명|번호|점수)$")
_DATE_SUFFIX_RE = re.compile(r"(일자|기일|일)$")

#: Event stems whose natural Korean modifier is "…된".
_PAST_EVENT_STEMS = (
    "체결", "접수", "수납", "지급", "위촉", "해촉", "해지", "심사", "완료",
    "개시", "개설", "폐쇄", "가입", "시작", "발생", "거래",
)
#: Stems whose natural modifier is forward-looking ("…되는").
_FUTURE_EVENT_STEMS = ("만기", "종료", "도래")

_OPERATORS: dict[str, str] = {">=": "이상", ">": "초과", "<=": "이하", "<": "미만"}
_COMPARATIVE: dict[str, str] = {">": "큰", ">=": "큰", "<": "작은", "<=": "작은"}

#: Aggregate → the Korean noun phrase an analyst would use.
_AGG_PREFIX: dict[str, tuple[str, ...]] = {
    "SUM": ("총", "합계"),
    "AVG": ("평균",),
    "MAX": ("최대", "가장 큰"),
    "MIN": ("최소", "가장 작은"),
}

_ENDINGS: tuple[str, ...] = (
    "알려줘.",
    "알려주세요.",
    "조회해줘.",
    "뽑아줘.",
    "보여줘.",
    "집계해줘.",
    "정리해줘.",
    "확인해줘.",
)
#: Interrogative endings attach to a noun phrase, so each carries the particle
#: pair rather than a fixed allomorph ("…5건이?" vs "…5개가?").
_QUESTION_ENDINGS: tuple[tuple[str, str], ...] = (
    ("은/는", "?"),
    ("이/가", " 어떻게 되나요?"),
    ("은/는", " 얼마나 되나요?"),
)


class BackTranslator:
    """Renders a :class:`~aegis_sql.flywheel.sql_sampler.SQLProgram` as Korean."""

    def __init__(
        self,
        schema: SchemaGraph,
        profile: SchemaProfile,
        glossary: Any = None,
        llm: LLMClient | None = None,
        registry: PromptRegistry | None = None,
        seed: int = 20260824,
    ) -> None:
        self.schema = schema
        self.profile = profile
        self.glossary: list[GlossaryEntry] = _as_entries(glossary)
        self.llm = llm
        self.registry = registry
        self.seed = int(seed)

        self._by_predicate: dict[str, GlossaryEntry] = {}
        self._by_column: dict[str, GlossaryEntry] = {}
        self._index_glossary()
        self._card: SchemaCardBuilder | None = None
        self._llm_failures = 0

    # -- public ------------------------------------------------------------ #

    def translate(self, program: SQLProgram) -> str:
        """One natural Korean question for ``program``; never raises."""
        rng = random.Random(f"{self.seed}:{program.template_id}:{program.key}")
        try:
            question = self._realise(program, rng)
        except Exception as exc:  # pragma: no cover - a bad slot must not kill a build
            log.warning("realisation failed", template=program.template_id, error=str(exc))
            question = self._fallback(program)
        question = _tidy(question)
        return self._rewrite(program, question) if self.llm is not None else question

    def translate_batch(self, programs: list[SQLProgram]) -> list[str]:
        out = [self.translate(p) for p in programs]
        if self.llm is not None:
            log.info("back-translated with llm", n=len(out), fallbacks=self._llm_failures)
        else:
            log.info("back-translated deterministically", n=len(out))
        return out

    # -- glossary indices -------------------------------------------------- #

    def _index_glossary(self) -> None:
        """Two lookups: exact predicate fragments, and column → owning term."""
        for entry in self.glossary:
            if entry.sql_hint:
                key = _predicate_key(entry.sql_hint)
                if key and key not in self._by_predicate:
                    self._by_predicate[key] = entry
            for qualified in entry.columns:
                # A term only *names* a column when it is unambiguous: either the
                # data dictionary agrees verbatim, or the term owns that column
                # alone.  Otherwise 손해율 would end up naming TB_CLM.PAY_AMT.
                col = self.schema.column(*qualified.split(".", 1)) if "." in qualified else None
                comment = (col.comment or "") if col else ""
                exact = entry.term == comment
                sole = len(entry.columns) == 1 and entry.term in comment
                if (exact or sole) and qualified not in self._by_column:
                    self._by_column[qualified] = entry

    # -- label resolution --------------------------------------------------- #

    def _noun(self, table: str) -> str:
        """Korean noun for one row of ``table`` (계약, 청구, 민원, …)."""
        info = self.schema.table(table)
        label = (info.comment if info else None) or table
        return label.split("/")[-1].strip()

    def _column_label(self, qualified: str) -> str:
        table, _, column = qualified.partition(".")
        col = self.schema.column(table, column)
        return (col.comment if col else None) or column

    def _dimension(self, qualified: str) -> str:
        """Dimension stem: 계약상태코드 → 계약상태, 지점명 → 지점."""
        stem = _LABEL_SUFFIX_RE.sub("", self._column_label(qualified))
        return stem or self._column_label(qualified)

    def _entity_of(self, qualified: str) -> str:
        """What a FK column counts: ``TB_CTRT.CUST_ID`` → 고객."""
        table, _, column = qualified.partition(".")
        col = self.schema.column(table, column)
        if col is not None and col.foreign_key is not None:
            return self._noun(col.foreign_key.to_table)
        return self._dimension(qualified)

    def _measure(self, qualified: str, rng: random.Random) -> str:
        """Business name of a measure — the 용어사전 term when it owns the column."""
        entry = self._by_column.get(qualified)
        if entry is not None:
            surfaces = _korean_surfaces(entry)
            if surfaces:
                return surfaces[rng.randrange(len(surfaces))]
        return self._column_label(qualified)

    def _value_surface(self, qualified: str, code: str, label: str, rng: random.Random) -> str:
        """Business name of a code value, preferring a 용어사전 alias of the predicate."""
        _, _, column = qualified.partition(".")
        entry = self._by_predicate.get(_predicate_key(f"{column} = '{code}'"))
        if entry is not None:
            surfaces = _korean_surfaces(entry)
            if surfaces:
                return surfaces[rng.randrange(len(surfaces))]
        return label

    # -- phrase builders ---------------------------------------------------- #

    def _when(self, slots: dict[str, Any], rng: random.Random, prefix: str = "date") -> str:
        """Verbalise a date window together with the event its column names."""
        kind = slots.get(f"{prefix}_kind")
        if not kind:
            return ""
        year, part, n = slots.get(f"{prefix}_year", 0), slots.get(f"{prefix}_part", 0), slots.get(f"{prefix}_n", 0)
        start, end = slots.get(f"{prefix}_start", ""), slots.get(f"{prefix}_end", "")

        if kind == "year":
            phrase = rng.choice((f"{year}년에", f"{year}년 한 해 동안", f"{year}년 기준으로"))
        elif kind == "half":
            half = "상반기" if part == 1 else "하반기"
            phrase = rng.choice((f"{year}년 {half}에", f"{year}년 {half} 동안"))
        elif kind == "quarter":
            phrase = rng.choice((f"{year}년 {part}분기에", f"{year}년 {part}분기 동안"))
        elif kind == "month":
            phrase = rng.choice((f"{year}년 {part}월에", f"{year}년 {part}월 한 달간"))
        elif kind == "recent":
            phrase = rng.choice((f"최근 {n}개월간", f"최근 {n}개월 동안", f"지난 {n}개월간"))
        else:
            phrase = f"{start[:4]}년 {int(start[4:6])}월부터 {end[:4]}년 {int(end[4:6])}월까지"

        column = slots.get(f"{prefix}_col", "")
        verb = self._event_verb(column) if column else ""
        return f"{phrase} {verb}".strip() if verb else f"{self._column_label(column)} 기준 {phrase}"

    def _event_verb(self, qualified: str) -> str:
        """계약체결일자 → 체결된, 만기일자 → 만기가 도래하는."""
        stem = _DATE_SUFFIX_RE.sub("", self._column_label(qualified))
        if stem.endswith(_PAST_EVENT_STEMS):
            return f"{stem}된"
        if stem.endswith(_FUTURE_EVENT_STEMS):
            return f"{stem}되는"
        return ""

    def _condition(self, slots: dict[str, Any], rng: random.Random) -> str:
        """A code-equality predicate as a Korean modifier ("실효된", "연납인")."""
        qualified, code = slots.get("filter_col"), slots.get("filter_value")
        if not qualified or code is None:
            return ""
        surface = self._value_surface(qualified, str(code), str(slots.get("filter_label") or code), rng)
        label = self._column_label(qualified)
        if "상태" in label:
            return rng.choice((f"{surface} 상태인", f"상태가 {surface}인"))
        if "채널" in label:
            channel = surface if surface.endswith("채널") else f"{surface} 채널"
            return f"{channel}을 통해 들어온" if "접수" in label else f"{channel}에서 모집된"
        if "유형" in label:
            return f"{surface} 유형의"
        if "등급" in label:
            return f"{surface} 등급인"
        if "방법" in label:
            return f"{josa(surface, '으로/로')} 납입된"
        if "결과" in label:
            return f"{josa(surface, '으로/로')} 심사된"
        if "주기" in label:
            return f"{surface}인"
        stem = self._dimension(qualified)
        return f"{josa(stem, '이/가')} {surface}인"

    def _metric(self, slots: dict[str, Any], rng: random.Random, entity: str) -> str:
        """The thing being asked for: 건수 / 합계 / 평균 / 비율 …"""
        agg = str(slots.get("agg") or "COUNT")
        measure = slots.get("measure")
        if agg == "COUNT":
            return f"{entity} {_count_noun(entity)}"
        if agg == "COUNT_DISTINCT":
            target = self._entity_of(str(slots.get("distinct_col", "")))
            return f"서로 다른 {target} 수" if rng.random() < 0.5 else f"{target} 수"
        if agg == "RATIO":
            condition = self._condition(slots, rng)
            return f"{condition} {entity} 비율".strip()
        if agg == "GROWTH":
            term = self._measure(str(measure), rng) if measure else entity
            return rng.choice((f"{term} 증감률", f"{term} 전년 대비 증가율"))
        term = self._measure(str(measure), rng) if measure else entity
        prefix = _AGG_PREFIX.get(agg, ("",))
        chosen = prefix[rng.randrange(len(prefix))]
        if chosen == "합계":
            return f"{term} 합계"
        return f"{chosen} {term}".strip()

    def _subject_metric(self, slots: dict[str, Any], rng: random.Random, entity: str) -> str:
        """:meth:`_metric` with the subject attached only when it is not implied."""
        metric = self._metric(slots, rng, entity)
        if str(slots.get("agg") or "COUNT") in {"COUNT", "COUNT_DISTINCT", "RATIO"}:
            return metric
        return f"{entity}의 {metric}"

    def _group(self, slots: dict[str, Any], rng: random.Random) -> str:
        qualified = slots.get("group_col")
        if not qualified:
            return ""
        stem = self._dimension(str(qualified))
        return rng.choice((f"{stem}별로", f"{stem}별", f"{stem} 기준으로"))

    def _threshold(self, slots: dict[str, Any], rng: random.Random) -> str:
        measure, op, value = slots.get("measure"), slots.get("op"), slots.get("threshold")
        if not measure or op is None or value is None:
            return ""
        term = self._measure(str(measure), rng)
        return f"{josa(term, '이/가')} {_quantity(self._column_label(str(measure)), int(value))} {_OPERATORS[op]}인"

    def _top(self, slots: dict[str, Any], rng: random.Random) -> str:
        limit = slots.get("limit")
        if not limit:
            return ""
        descending = str(slots.get("order", "DESC")).upper() == "DESC"
        if descending:
            return rng.choice((f"상위 {limit}개", f"많은 순으로 {limit}개"))
        return rng.choice((f"하위 {limit}개", f"적은 순으로 {limit}개"))

    def _having(self, slots: dict[str, Any], entity: str) -> str:
        k = slots.get("having_k")
        if not k:
            return ""
        count = f"{entity} {_count_noun(entity)}"
        return f"{count}{particle(count, '이/가')} {k}{_counter(entity)} 이상인"

    # -- realisation --------------------------------------------------------- #

    def _realise(self, program: SQLProgram, rng: random.Random) -> str:
        handler = getattr(self, f"_r_{program.template_id}", None)
        if handler is None:  # pragma: no cover - registry drift guard
            return self._fallback(program)
        return handler(program.slots, rng)

    def _fallback(self, program: SQLProgram) -> str:
        """Last resort: never emit an empty question."""
        nouns = " ".join(self._noun(t) for t in program.tables[:2])
        return f"{nouns} 관련 집계를 알려줘."

    def _end(self, body: str, rng: random.Random) -> str:
        """Attach a sentence-final request form to a noun phrase."""
        if rng.random() < 0.18:
            pair, tail = _QUESTION_ENDINGS[rng.randrange(len(_QUESTION_ENDINGS))]
            return f"{body}{particle(body, pair)}{tail}"
        ending = _ENDINGS[rng.randrange(len(_ENDINGS))]
        return f"{body}{particle(body, '을/를')} {ending}"

    # -- easy ---------------------------------------------------------------- #

    def _r_easy_count_by_code(self, slots: dict[str, Any], rng: random.Random) -> str:
        entity = self._noun(str(slots["fact"]))
        condition = self._condition(slots, rng)
        return self._end(f"{condition} {entity} {_count_noun(entity)}", rng)

    def _r_easy_count_in_window(self, slots: dict[str, Any], rng: random.Random) -> str:
        entity = self._noun(str(slots["fact"]))
        return self._end(f"{self._when(slots, rng)} {entity} {_count_noun(entity)}", rng)

    def _r_easy_aggregate_in_window(self, slots: dict[str, Any], rng: random.Random) -> str:
        entity = self._noun(str(slots["fact"]))
        metric = self._metric(slots, rng, entity)
        when = self._when(slots, rng)
        body = rng.choice((f"{when} {entity}의 {metric}", f"{when} {entity} 기준 {metric}"))
        return self._end(_tidy(body), rng)

    def _r_easy_aggregate_by_code(self, slots: dict[str, Any], rng: random.Random) -> str:
        entity = self._noun(str(slots["fact"]))
        return self._end(f"{self._condition(slots, rng)} {self._subject_metric(slots, rng, entity)}", rng)

    def _r_easy_count_over_threshold(self, slots: dict[str, Any], rng: random.Random) -> str:
        entity = self._noun(str(slots["fact"]))
        return self._end(f"{self._threshold(slots, rng)} {entity} {_count_noun(entity)}", rng)

    def _r_easy_distinct_count(self, slots: dict[str, Any], rng: random.Random) -> str:
        entity = self._noun(str(slots["fact"]))
        target = self._entity_of(str(slots["distinct_col"]))
        when = self._when(slots, rng)
        body = rng.choice(
            (
                f"{when} {entity}의 {target} 수",
                f"{when} {entity}에 나타난 서로 다른 {target} 수",
            )
        )
        return self._end(body, rng)

    def _r_easy_top_n_rows(self, slots: dict[str, Any], rng: random.Random) -> str:
        entity = self._noun(str(slots["fact"]))
        term = self._measure(str(slots["measure"]), rng)
        when = self._when(slots, rng)
        limit, order = slots.get("limit"), str(slots.get("order", "DESC")).upper()
        adjective = "높은" if order == "DESC" else "낮은"
        body = rng.choice(
            (
                f"{when} {entity} 중 {term}{particle(term, '이/가')} 가장 {adjective} {limit}건",
                f"{when} {entity} 중 {term} {adjective} 순으로 {limit}건",
            )
        )
        return self._end(_tidy(body), rng)

    # -- medium -------------------------------------------------------------- #

    def _r_medium_group_by_dimension(self, slots: dict[str, Any], rng: random.Random) -> str:
        entity = self._noun(str(slots["fact"]))
        metric = self._subject_metric(slots, rng, entity)
        body = f"{self._when(slots, rng)} {self._group(slots, rng)} {metric} {self._top(slots, rng)}"
        return self._end(_tidy(body), rng)

    def _r_medium_group_by_code_label(self, slots: dict[str, Any], rng: random.Random) -> str:
        entity = self._noun(str(slots["fact"]))
        stem = self._dimension(str(slots["group_col"]))
        metric = self._metric(slots, rng, entity)
        body = rng.choice(
            (
                f"{self._when(slots, rng)} {stem}별 {metric}",
                f"{self._when(slots, rng)} {stem} 이름과 함께 집계한 {metric}",
            )
        )
        return self._end(_tidy(body), rng)

    def _r_medium_date_bucket_trend(self, slots: dict[str, Any], rng: random.Random) -> str:
        entity = self._noun(str(slots["fact"]))
        unit = "월별" if slots.get("bucket") == "month" else "연도별"
        extra = ""
        if slots.get("measure"):
            extra = f"와 {self._measure(str(slots['measure']), rng)} 합계"
        body = f"{self._when(slots, rng)} {entity} {_count_noun(entity)}{extra}의 {unit} 추이"
        return self._end(_tidy(body), rng)

    def _r_medium_having_threshold(self, slots: dict[str, Any], rng: random.Random) -> str:
        entity = self._noun(str(slots["fact"]))
        stem = self._dimension(str(slots["group_col"]))
        body = f"{self._when(slots, rng)} {stem}별 {self._having(slots, entity)} {stem}"
        return self._end(_tidy(body), rng)

    def _r_medium_multi_predicate(self, slots: dict[str, Any], rng: random.Random) -> str:
        entity = self._noun(str(slots["fact"]))
        when = self._when(slots, rng)
        rest = [self._condition(slots, rng), self._threshold(slots, rng)]
        rng.shuffle(rest)
        parts = [when, *rest] if rng.random() < 0.7 else [*rest, when]
        body = " ".join([*parts, self._subject_metric(slots, rng, entity)])
        return self._end(_tidy(body), rng)

    def _r_medium_ratio_nullif(self, slots: dict[str, Any], rng: random.Random) -> str:
        entity = self._noun(str(slots["fact"]))
        condition = self._condition(slots, rng)
        stem = self._dimension(str(slots["group_col"]))
        body = rng.choice(
            (
                f"{self._when(slots, rng)} {stem}별 전체 {entity} 대비 {condition} {entity}의 비중",
                f"{self._when(slots, rng)} {stem}별 {condition} {entity} 비율",
            )
        )
        return self._end(_tidy(body), rng)

    def _r_medium_three_table_join(self, slots: dict[str, Any], rng: random.Random) -> str:
        entity = self._noun(str(slots["fact"]))
        metric = self._metric(slots, rng, entity)  # subject is already explicit
        body = (
            f"{self._when(slots, rng)} {entity}{particle(entity, '을/를')} "
            f"{self._dimension(str(slots['group_col']))}별로 묶었을 때의 {metric} {self._top(slots, rng)}"
        )
        return self._end(_tidy(body), rng)

    # -- hard ----------------------------------------------------------------- #

    def _r_hard_above_global_average(self, slots: dict[str, Any], rng: random.Random) -> str:
        entity = self._noun(str(slots["fact"]))
        term = self._measure(str(slots["measure"]), rng)
        direction = _COMPARATIVE[str(slots["op"])]
        body = (
            f"{self._when(slots, rng)} {entity} 중에서 {term}{particle(term, '이/가')} "
            f"전체 평균보다 {direction} {entity} {_count_noun(entity)}"
        )
        return self._end(_tidy(body), rng)

    def _r_hard_cte_two_stage(self, slots: dict[str, Any], rng: random.Random) -> str:
        term = self._measure(str(slots["measure"]), rng)
        stem = self._dimension(str(slots["group_col"]))
        body = (
            f"{self._when(slots, rng)} {stem}별 {term} 합계를 구한 뒤, 그 값이 전체 평균보다 높은 "
            f"{stem} 상위 {slots.get('limit')}개"
        )
        return self._end(_tidy(body), rng)

    def _r_hard_period_over_period(self, slots: dict[str, Any], rng: random.Random) -> str:
        term = self._measure(str(slots["measure"]), rng)
        stem = self._dimension(str(slots["group_col"]))
        current, previous = self._period(slots, "date"), self._period(slots, "prev")
        body = rng.choice(
            (
                f"{stem}별로 {previous} 대비 {current} {term} 합계 증감률",
                f"{current}{particle(current, '과/와')} {previous}의 {stem}별 {term} 합계를 비교한 증감률",
            )
        )
        return self._end(_tidy(body), rng)

    def _r_hard_anti_join_not_exists(self, slots: dict[str, Any], rng: random.Random) -> str:
        entity = self._noun(str(slots["fact"]))
        child = self._noun(str(slots["child"]))
        body = rng.choice(
            (
                f"{self._when(slots, rng)} {entity} 중 {child} 이력이 한 번도 없는 "
                f"{entity} {_count_noun(entity)}",
                f"{self._when(slots, rng)} {entity} 가운데 {child}{particle(child, '이/가')} "
                f"전혀 발생하지 않은 {entity} {_count_noun(entity)}",
            )
        )
        return self._end(_tidy(body), rng)

    def _r_hard_left_join_is_null(self, slots: dict[str, Any], rng: random.Random) -> str:
        entity = self._noun(str(slots["fact"]))
        child = self._noun(str(slots["child"]))
        body = (
            f"{self._when(slots, rng)} {child} 실적이 전혀 없는 {entity}{particle(entity, '을/를')} "
            f"{self._group(slots, rng)} 집계한 결과"
        )
        return self._end(_tidy(body), rng)

    def _r_hard_nested_aggregate(self, slots: dict[str, Any], rng: random.Random) -> str:
        term = self._measure(str(slots["measure"]), rng)
        stem = self._dimension(str(slots["group_col"]))
        unit = self._noun(str(slots["entity"]))
        outer = "가장 큰 값" if str(slots.get("agg")) == "MAX" else "평균"
        body = (
            f"{self._when(slots, rng)} {unit}별 {term} 합계를 먼저 구한 다음, "
            f"{stem}별로 그 {outer} 기준 상위 {slots.get('limit')}개"
        )
        return self._end(_tidy(body), rng)

    def _r_hard_rank_emulation(self, slots: dict[str, Any], rng: random.Random) -> str:
        term = self._measure(str(slots["measure"]), rng)
        stem = self._dimension(str(slots["group_col"]))
        limit = slots.get("limit")
        body = rng.choice(
            (
                f"{self._when(slots, rng)} {stem}별 {term} 합계에 순위를 매겨 상위 {limit}개",
                f"{self._when(slots, rng)} {term} 합계가 많은 {stem} 순위 상위 {limit}개",
            )
        )
        return self._end(_tidy(body), rng)

    def _r_hard_join_having_subquery(self, slots: dict[str, Any], rng: random.Random) -> str:
        entity = self._noun(str(slots["fact"]))
        term = self._measure(str(slots["measure"]), rng)
        stem = self._dimension(str(slots["group_col"]))
        sub_stem = self._dimension(str(slots["sub_col"]))
        sub_label = str(slots.get("sub_label") or slots.get("sub_value"))
        body = (
            f"{self._when(slots, rng)} {sub_stem}{particle(sub_stem, '이/가')} {sub_label}인 {entity} 가운데 "
            f"{self._having(slots, entity)} {stem}별 {_count_noun(entity)}와 {term} 합계"
        )
        return self._end(_tidy(body), rng)

    def _period(self, slots: dict[str, Any], prefix: str) -> str:
        """Bare period name for comparison sentences (no event verb)."""
        kind, year, part = slots.get(f"{prefix}_kind"), slots.get(f"{prefix}_year", 0), slots.get(f"{prefix}_part", 0)
        if kind == "half":
            return f"{year}년 {'상반기' if part == 1 else '하반기'}"
        if kind == "quarter":
            return f"{year}년 {part}분기"
        if kind == "month":
            return f"{year}년 {part}월"
        return f"{year}년"

    # -- optional LLM rewriting --------------------------------------------- #

    def _rewrite(self, program: SQLProgram, fallback: str) -> str:
        """Ask the LLM for a more natural surface; keep ``fallback`` unless it wins."""
        if self.registry is None or BACKTRANSLATE_PROMPT not in self.registry:
            return fallback
        try:
            if self._card is None:
                self._card = SchemaCardBuilder(self.schema, self.profile)
            card = self._card.render(LinkedSchema(tables=list(program.tables)), style="compact")
            prompt = self.registry.render(BACKTRANSLATE_PROMPT, schema_card=card, sql=program.sql)
            response = self.llm.complete([Message(role="user", content=prompt)])  # type: ignore[union-attr]
            candidate = _first_line(response.text)
        except Exception as exc:
            self._llm_failures += 1
            log.debug("llm back-translation failed", template=program.template_id, error=str(exc))
            return fallback
        if not _acceptable(candidate, program):
            self._llm_failures += 1
            return fallback
        return candidate


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _as_entries(glossary: Any) -> list[GlossaryEntry]:
    if glossary is None:
        return []
    entries = getattr(glossary, "entries", glossary)
    return [e for e in entries if isinstance(e, GlossaryEntry)]


def _korean_surfaces(entry: GlossaryEntry) -> list[str]:
    """Term plus aliases, minus ASCII-only and digit-bearing surfaces."""
    return [
        s
        for s in dict.fromkeys((entry.term, *entry.aliases))
        if _HANGUL_RE.search(s) and not any(c.isdigit() for c in s)
    ]


def _predicate_key(fragment: str) -> str:
    """Normalise ``CTRT_STAT_CD = '02'`` so a hint and a slot binding compare equal."""
    return re.sub(r"\s+", "", fragment).upper()


def _counter(entity: str) -> str:
    """Korean counter word for a noun: 고객/설계사 take 명, everything else 건."""
    return "명" if entity in {"고객", "설계사", "직원", "사람"} else "건"


def _count_noun(entity: str) -> str:
    """How the count itself is named: "설계사 수" but "계약 건수"."""
    return "수" if _counter(entity) == "명" else "건수"


def _quantity(label: str, value: int) -> str:
    """Render a threshold in the unit its column implies."""
    if re.search(r"금액|보험료|료$|액$", label):
        return _amount(value)
    if "점수" in label or "만족도" in label:
        return f"{value}점"
    if "일수" in label:
        return f"{value}일"
    if "년수" in label or "기간" in label:
        return f"{value}년"
    if "직원수" in label or "인원" in label:
        return f"{value}명"
    if label.endswith("수") or "횟수" in label:
        return f"{value}건"
    return f"{value:,}"


def _amount(value: int) -> str:
    """KRW in the myriad units Korean actually speaks (1억 5000만원)."""
    if value <= 0:
        return f"{value}원"
    eok, remainder = divmod(value, 100_000_000)
    man, won = divmod(remainder, 10_000)
    parts: list[str] = []
    if eok:
        parts.append(f"{eok}억")
    if man:
        parts.append(f"{man}만")
    if won or not parts:
        parts.append(f"{won:,}")
    return " ".join(parts) + "원"


def _tidy(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").replace(" ,", ",")).strip()


def _first_line(text: str) -> str:
    for line in (text or "").splitlines():
        stripped = line.strip().strip("`").strip()
        if stripped:
            return stripped
    return ""


def _acceptable(candidate: str, program: SQLProgram) -> bool:
    """Reject an LLM rewrite that is empty, verbose, or leaks physical names."""
    if not candidate or not _HANGUL_RE.search(candidate) or len(candidate) > 220:
        return False
    identifiers = {t.upper() for t in program.tables}
    identifiers |= {str(v).split(".")[-1].upper() for k, v in program.slots.items() if k.endswith("_col")}
    return not any(ident in _IDENT_RE.findall(candidate.upper()) for ident in identifiers)
