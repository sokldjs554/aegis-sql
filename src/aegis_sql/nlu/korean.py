"""Deterministic Korean question understanding for a legacy-schema Text-to-SQL engine.

Three properties of Korean finance questions make a rule layer worth more than one
more LLM call:

1. **Dates are strings.**  Every date column in the target core is ``CHAR(8)``
   ``'YYYYMMDD'`` text (see ``data/demo/schema.sql``), so ``"작년 하반기"`` has to
   become ``BETWEEN '20250701' AND '20251231'`` *before* generation.  Resolving the
   ~40 relative expressions Korean analysts actually type is deterministic, unit
   testable and free.  Delegating it to a model re-introduces a silent failure class
   that execution checks cannot catch: a query with the wrong year still runs, still
   returns rows, and still looks right.
2. **Korean numerals are myriad-grouped.**  ``1억5천만`` is ``1·10^8 + 5000·10^4``,
   and ``3.5억`` is not ``35만``.  The whole ``조/억/만/천/백/십`` lattice is folded
   into a single integer here so that predicates compare against real KRW values.
3. **Korean is agglutinative.**  ``지점별로`` / ``지점의`` / ``지점에서`` are one
   lexical key for schema linking.  Stripping 조사 (particles) while *keeping* the raw
   surface lifts linking recall without a morphological analyser — which this system
   cannot ship anyway, since no model may be downloaded at runtime.

Everything below is pure given ``self.today``; the only clock read is the constructor
default, so tests pin the reference date and get byte-identical output.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

from aegis_sql.observability.logging import get_logger
from aegis_sql.types import NormalizedQuestion

log = get_logger("nlu.korean")

#: Cue flags always present in ``entities["cues"]`` (the router reads them positionally).
CUE_NAMES: tuple[str, ...] = (
    "aggregate",
    "ranking",
    "temporal",
    "comparison",
    "nested",
    "ratio",
    "set_op",
)

# --------------------------------------------------------------------------- #
# Tokenisation
# --------------------------------------------------------------------------- #

#: Script-boundary tokenizer: a run of digits, Hangul, or Latin is one token, so
#: ``"20만원이상인"`` → ``20`` / ``만원이상인`` and ``"VIP고객의"`` → ``VIP`` / ``고객의``.
_SCAN_RE = re.compile(
    r"(?P<num>\d[\d,]*(?:\.\d+)?)"
    r"|(?P<han>[가-힣ㄱ-ㅎㅏ-ㅣ]+)"
    r"|(?P<lat>[A-Za-z]+)"
    r"|(?P<sym>[%℃]+)"
)

#: 조사.  Longest-first so ``"에게"`` wins over ``"에"``.
_PARTICLES: tuple[str, ...] = tuple(
    sorted(
        {
            "은", "는", "이", "가", "을", "를", "의", "에", "에서", "에게", "으로", "로",
            "와", "과", "랑", "이랑", "도", "만", "부터", "까지", "보다", "처럼", "한테",
            "께", "이나", "나", "든지", "라도", "마다", "조차", "밖에", "뿐", "대로",
        },
        key=len,
        reverse=True,
    )
)

_LATIN_STOPWORDS = frozenset({"top", "best", "and", "or", "the", "vs", "by", "of", "in"})

_STOPWORDS: frozenset[str] = frozenset(
    {
        # 질의 상투어 / 기능어
        "알려줘", "알려", "보여줘", "보여", "조회", "검색", "확인", "구해줘", "뽑아줘", "출력",
        "리스트", "목록", "얼마", "무엇", "어떤", "어느", "누구", "언제", "어디", "어떻게",
        "각각", "그리고", "또는", "혹은", "제외", "제외하고", "포함", "기준", "경우", "관련",
        "대한", "대해", "위한", "통해", "해당", "이번", "저번", "지난", "다음", "그것", "이것",
        "입니다", "인가요", "일까요", "해줘", "정렬", "순서", "따라", "함께", "모두", "각",
        # 분석 상투어
        "합계", "합산", "총계", "총액", "전체", "평균", "최대", "최소", "개수", "건수", "비율",
        "비중", "퍼센트", "상위", "하위", "순위", "랭킹", "최다", "가장", "제일", "추이",
        "증감", "추세", "현황", "분포", "통계", "그룹", "기간", "시점", "이상", "이하",
        "초과", "미만", "이내", "사이", "부터", "까지", "보다", "대비", "넘는", "넘게",
        "동안", "높은", "낮은", "많은", "적은", "좋은", "나쁜", "있는", "없는", "여부",
        "내역", "정보", "대상", "결과", "왜",
        # 시간 표현 (date_range 로 이미 구조화되었으므로 값 후보가 아니다)
        "오늘", "금일", "어제", "그제", "올해", "작년", "금년", "전년", "재작년", "지난해",
        "이번주", "금주", "지난주", "전주", "이번달", "이달", "지난달", "금월", "전월",
        "당월", "상반기", "하반기", "분기", "전분기", "연초", "연도", "년도", "개월",
    }
)


#: Digits whose Korean reading ends in a consonant (영·일·삼·육·칠·팔).
_BATCHIM_DIGITS = frozenset("013678")


def josa(word: str, pair: str) -> str:
    """Pick the 받침-correct particle: ``josa("건수", "을/를")`` → ``"를"``.

    Sub-questions and clarifying questions are shown to users verbatim, and
    ``"건수을(를)"`` reads like a broken template.  Hangul syllables encode their
    final consonant positionally (``(code - 0xAC00) % 28``), so this is exact for
    Korean and falls back to the vowel form for Latin.
    """
    with_batchim, without = pair.split("/")
    # Skip trailing punctuation so "TB_CTRT(계약)" agrees on 약, not on ")".
    core = word.rstrip(")]}>」』\"' .")
    if not core:
        return without
    last = core[-1]
    if "가" <= last <= "힣":
        return with_batchim if (ord(last) - 0xAC00) % 28 else without
    if last.isdigit():
        return with_batchim if last in _BATCHIM_DIGITS else without
    return without


def _normalize_text(text: str) -> str:
    """NFKC-fold (full-width → ASCII) and collapse all whitespace runs to one space."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


def _strip_particle(token: str) -> str | None:
    """Return the stem when a trailing 조사 can be removed safely, else ``None``.

    The ``len(stem) >= 2`` guard is what keeps this from mangling real words:
    ``"미만"`` would otherwise become ``"미"`` and ``"하나"`` would become ``"하"``.
    """
    for particle in _PARTICLES:
        if len(token) > len(particle) and token.endswith(particle):
            stem = token[: -len(particle)]
            if len(stem) >= 2:
                return stem
    return None


# --------------------------------------------------------------------------- #
# Korean numerals
# --------------------------------------------------------------------------- #

_MAGNITUDE: dict[str, int] = {"조": 10**12, "억": 10**8, "만": 10**4}
_MINOR: dict[str, int] = {"천": 1000, "백": 100, "십": 10}
_UNIT_CHARS = "조억만천백십"

_NUM = r"\d[\d,]*(?:\.\d+)?"
#: One or more ``<숫자><단위?>`` chunks; a bare unit (``천만원``) is also a chunk.
_AMOUNT_BODY = rf"(?:(?:{_NUM})\s*[{_UNIT_CHARS}]?|[{_UNIT_CHARS}])+"
_AMOUNT_RE = re.compile(rf"(?<![\d.,]){_AMOUNT_BODY}\s*원?")
_CHUNK_RE = re.compile(
    rf"(?P<num>{_NUM})\s*(?P<unit>[{_UNIT_CHARS}])?|(?P<solo>[{_UNIT_CHARS}])"
)

#: Counters that prove a number is *not* money (``20만명``, ``3개월``, ``2025년``).
_NON_MONETARY = set("년월일주분회차세명건개번위등점시초인대장권통%")


def parse_korean_number(surface: str) -> int | None:
    """Fold a Korean numeral surface into an integer.

    ``1억5천만`` → ``150000000``, ``3.5억`` → ``350000000``, ``천만`` → ``10000000``.
    A magnitude unit (``조/억/만``) closes the current section and multiplies it, which
    is exactly why ``5천만`` is ``5000 * 10^4`` and not ``5000 + 10^4``.
    """
    total = 0.0
    section = 0.0
    seen = False
    for m in _CHUNK_RE.finditer(surface):
        raw, unit = m.group("num"), m.group("unit") or m.group("solo")
        if raw is None and unit is None:
            continue
        seen = True
        value = float(raw.replace(",", "")) if raw else 1.0
        if unit is None:
            section += value
        elif unit in _MINOR:
            section += value * _MINOR[unit]
        else:
            if raw is not None:
                section += value
            if section == 0.0:
                section = 1.0
            total += section * _MAGNITUDE[unit]
            section = 0.0
    if not seen:
        return None
    return int(round(total + section))


def _amount_is_monetary(surface: str, text: str, end: int) -> bool:
    """Filter the numeral matcher down to things that are plausibly KRW amounts."""
    if surface.endswith("원"):
        return True
    following = text[end] if end < len(text) else ""
    if following in _NON_MONETARY:
        return False
    if any(ch in _UNIT_CHARS for ch in surface):
        return True
    # A bare number: only large, explicitly-written figures count, and a lone
    # four-digit number in the calendar range is far more likely to be a year.
    digits = surface.replace(",", "")
    if "," in surface:
        return True
    return digits.isdigit() and len(digits) >= 4 and not (len(digits) == 4 and 1900 <= int(digits) <= 2100)


# --------------------------------------------------------------------------- #
# Date arithmetic helpers
# --------------------------------------------------------------------------- #


def _fmt(day: date) -> str:
    return day.strftime("%Y%m%d")


def _last_day(year: int, month: int) -> int:
    if month == 12:
        return 31
    return (date(year, month + 1, 1) - timedelta(days=1)).day


def _add_months(day: date, delta: int) -> date:
    """Month arithmetic with day clamping (``2026-03-31`` minus one month → ``2026-02-28``)."""
    total = (day.year * 12 + day.month - 1) + delta
    year, month = divmod(total, 12)
    month += 1
    return date(year, month, min(day.day, _last_day(year, month)))


def _year_range(year: int) -> tuple[str, str]:
    return f"{year}0101", f"{year}1231"


def _month_range(year: int, month: int) -> tuple[str, str]:
    return f"{year}{month:02d}01", f"{year}{month:02d}{_last_day(year, month):02d}"


def _quarter_range(year: int, quarter: int) -> tuple[str, str]:
    first = 3 * (quarter - 1) + 1
    last = first + 2
    return f"{year}{first:02d}01", f"{year}{last:02d}{_last_day(year, last):02d}"


def _half_range(year: int, half: str) -> tuple[str, str]:
    return (f"{year}0101", f"{year}0630") if half == "상반기" else (f"{year}0701", f"{year}1231")


def _week_range(day: date) -> tuple[str, str]:
    monday = day - timedelta(days=day.weekday())
    return _fmt(monday), _fmt(monday + timedelta(days=6))


# --------------------------------------------------------------------------- #
# Date expression patterns (evaluated in this order; first match owns the span)
# --------------------------------------------------------------------------- #

_REL_YEAR = r"지지난해|재작년|전년도|지난해|작년|전년|올해|금년|당해"
_REL_YEAR_OFFSET: dict[str, int] = {
    "지지난해": -2, "재작년": -2, "전년도": -1, "지난해": -1, "작년": -1,
    "전년": -1, "올해": 0, "금년": 0, "당해": 0,
}

_P_YMD_KOR = re.compile(r"(?<!\d)(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일")
_P_YMD_SEP = re.compile(r"(?<!\d)(\d{4})[-./](\d{1,2})[-./](\d{1,2})(?!\d)")
_P_YMD_FLAT = re.compile(r"(?<!\d)(\d{8})(?!\d)")
_P_Y_HALF = re.compile(r"(?<!\d)(\d{4})\s*년?\s*(상반기|하반기)")
_P_Y_QTR = re.compile(r"(?<!\d)(\d{4})\s*년\s*([1-4])\s*/?\s*분기")
_P_YM_KOR = re.compile(r"(?<!\d)(\d{4})\s*년\s*(\d{1,2})\s*월")
_P_YM_SEP = re.compile(r"(?<!\d)(\d{4})[-./](\d{1,2})(?![-./\d])")
_P_REL_HALF = re.compile(rf"({_REL_YEAR})\s*(상반기|하반기)")
_P_REL_QTR = re.compile(rf"({_REL_YEAR})\s*([1-4])\s*/?\s*분기")
_P_REL_MONTH = re.compile(rf"({_REL_YEAR})\s*(\d{{1,2}})\s*월")
_P_YEAR = re.compile(r"(?<!\d)(\d{4})\s*년도?(?!\s*\d)")
_P_RECENT = re.compile(r"(?:최근|지난|직전|최종)\s*(\d{1,3})\s*(개월|일|주일|주|달|년|분기)")
_P_YTD = re.compile(r"연초\s*(?:부터|이후)|올해\s*들어|금년\s*들어|올해\s*초부터|올들어|금년\s*초부터")
_P_THIS_WEEK = re.compile(r"이번\s*주(?!\s*[일말])|금주")
_P_LAST_WEEK = re.compile(r"지난\s*주(?!\s*[일말])|저번\s*주|전주")
_P_THIS_MONTH = re.compile(r"이번\s*달|이달|금월|당월")
_P_LAST_MONTH = re.compile(r"지난\s*달|저번\s*달|전월|전달")
_P_LAST_QTR = re.compile(r"전분기|지난\s*분기|직전\s*분기")
_P_THIS_QTR = re.compile(r"이번\s*분기|금분기|당분기")
_P_QTR = re.compile(r"(?<!\d)([1-4])\s*/?\s*분기")
_P_HALF = re.compile(r"(상반기|하반기)")
_P_REL_YEAR_ONLY = re.compile(rf"({_REL_YEAR})")
_P_TODAY = re.compile(r"오늘|금일")
_P_YESTERDAY = re.compile(r"어제|어저께|전일")
_P_DAY_BEFORE = re.compile(r"그저께|그제|그끄제")
_P_MONTH = re.compile(r"(?<!\d)(\d{1,2})\s*월(?!\s*\d)")


# --------------------------------------------------------------------------- #
# Cue / entity patterns
# --------------------------------------------------------------------------- #

_CUE_RE: dict[str, re.Pattern[str]] = {
    "aggregate": re.compile(
        r"합계|합산|총액|총계|총\s|총건|전체|평균|최대|최소|개수|건수|몇\s*건|몇\s*개|"
        r"(?<![가-힣])명|인원|비율|비중|퍼센트|%|카운트|누적"
    ),
    "ranking": re.compile(
        r"상위|하위|최다|최소|가장|제일|베스트|순위|랭킹|top\s*\d|top\d|"
        r"\d+\s*(?:개|건|명)\s*만",
        re.IGNORECASE,
    ),
    "temporal": re.compile(r"추이|월별|연도별|년도별|분기별|일별|주별|기간별|시계열|추세|증감"),
    "comparison": re.compile(
        r"이상|이하|초과|미만|넘는|넘게|보다\s*(?:더\s*)?(?:많|적|큰|작|높|낮)|사이|이내|범위"
    ),
    "nested": re.compile(
        r"각각|별로|평균\s*보다|평균보다|전체\s*평균|전사\s*평균|보다\s*(?:더\s*)?(?:높|큰|많)|"
        r"대비|중에서|가운데"
    ),
    "ratio": re.compile(r"비율|비중|퍼센트|%|률|율|대비|(?:인|건|명|계약|고객|월|년)\s*당(?![가-힣])"),
    "set_op": re.compile(r"그리고\s*동시에|동시에|또는|혹은|제외하고|제외한|아닌|아니면서"),
}

_P_TOP_K = (
    re.compile(r"(?:상위|하위|베스트|랭킹|순위)\s*(\d{1,4})\s*(?:개|건|명|위|가지)?"),
    re.compile(r"(?:top|best)\s*(\d{1,4})", re.IGNORECASE),
    re.compile(r"(?<!\d)(\d{1,4})\s*(?:개|건|명|가지)\s*만"),
)

_CMP_UNIT = r"(?:원|건|명|개|세|점|%|퍼센트|회|년|일)?"
_P_BETWEEN = re.compile(
    rf"(?<![\d.,])(?P<a>{_AMOUNT_BODY})\s*{_CMP_UNIT}\s*(?:부터|에서|~|-)\s*"
    rf"(?P<b>{_AMOUNT_BODY})\s*{_CMP_UNIT}\s*(?:까지|사이|이내)"
)
_P_CMP_SUFFIX = re.compile(
    rf"(?<![\d.,])(?P<val>{_AMOUNT_BODY})\s*{_CMP_UNIT}\s*"
    r"(?P<op>이상인|이하인|미만인|초과하는|이상|이하|초과|미만|넘는|넘게)"
)
_P_CMP_BODA = re.compile(
    rf"(?<![\d.,])(?P<val>{_AMOUNT_BODY})\s*{_CMP_UNIT}\s*보다\s*(?:더\s*)?"
    r"(?P<op>많|큰|높|적|작|낮)"
)
_CMP_OPS: dict[str, str] = {
    "이상": ">=", "이상인": ">=", "이하": "<=", "이하인": "<=",
    "초과": ">", "초과하는": ">", "넘는": ">", "넘게": ">",
    "미만": "<", "미만인": "<",
    "많": ">", "큰": ">", "높": ">", "적": "<", "작": "<", "낮": "<",
}

_P_GROUPBY = re.compile(r"([가-힣A-Za-z]{1,6})별")
#: ``별`` is part of the noun here, not a group-by marker.
_BYEOL_EXCLUDE = frozenset({"개", "특", "각", "차", "구", "이", "작", "송", "선", "판", "성별"})
#: Stems where the ``별`` belongs to the dimension name itself (성별 = gender).
_BYEOL_KEEP_WHOLE = frozenset({"성"})

_P_QUOTED = re.compile(r"[\"'“‘「『]([^\"'”’」』\n]{1,40})[\"'”’」』]")

_INTENT_RE: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ratio", re.compile(r"비율|비중|퍼센트|률|율|(?:인|건|명|계약|고객)\s*당(?![가-힣])")),
    ("rank", re.compile(r"상위|하위|순위|랭킹|최다|가장|제일|베스트|top\s*\d", re.IGNORECASE)),
    # ``평균보다`` is a *comparison against* an average, not a request for one.
    ("avg", re.compile(r"평균(?!\s*(?:보다|대비|이상|이하|초과|미만))")),
    ("max", re.compile(r"최대|최고|가장\s*큰|가장\s*많은")),
    ("min", re.compile(r"최저|가장\s*작은|가장\s*적은")),
    ("count", re.compile(r"건수|개수|몇\s*건|몇\s*개|몇\s*명|인원|카운트|(?<![가-힣])명\s*수")),
    ("sum", re.compile(r"합계|합산|총액|총계|누적|총\s|얼마")),
)


def _free(mask: bytearray, start: int, end: int) -> bool:
    return not any(mask[start:end])


def _mark(mask: bytearray, start: int, end: int) -> None:
    for i in range(start, end):
        mask[i] = 1


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


class KoreanNormalizer:
    """Rule-based Korean question analyser — no external NLP dependency.

    ``today`` is injectable precisely because every relative date expression is
    resolved against it; pinning it makes the whole pipeline reproducible.
    """

    __slots__ = ("today", "_date_rules")

    def __init__(self, today: date | str | None = None) -> None:
        self.today: date = _coerce_today(today)
        self._date_rules: tuple[tuple[re.Pattern[str], Callable[[re.Match[str]], _Resolved | None]], ...] = (
            (_P_YMD_KOR, self._r_ymd_kor),
            (_P_YMD_SEP, self._r_ymd_sep),
            (_P_YMD_FLAT, self._r_ymd_flat),
            (_P_Y_QTR, self._r_y_qtr),
            (_P_Y_HALF, self._r_y_half),
            (_P_YM_KOR, self._r_ym_kor),
            (_P_YM_SEP, self._r_ym_sep),
            (_P_REL_QTR, self._r_rel_qtr),
            (_P_REL_HALF, self._r_rel_half),
            (_P_REL_MONTH, self._r_rel_month),
            (_P_YEAR, self._r_year),
            (_P_RECENT, self._r_recent),
            (_P_YTD, self._r_ytd),
            (_P_THIS_WEEK, self._r_this_week),
            (_P_LAST_WEEK, self._r_last_week),
            (_P_THIS_MONTH, self._r_this_month),
            (_P_LAST_MONTH, self._r_last_month),
            (_P_LAST_QTR, self._r_last_qtr),
            (_P_THIS_QTR, self._r_this_qtr),
            (_P_QTR, self._r_qtr),
            (_P_HALF, self._r_half),
            (_P_REL_YEAR_ONLY, self._r_rel_year),
            (_P_TODAY, self._r_today),
            (_P_YESTERDAY, self._r_yesterday),
            (_P_DAY_BEFORE, self._r_day_before),
            (_P_MONTH, self._r_month),
        )

    # -- tokenisation ----------------------------------------------------- #

    def tokenize(self, text: str) -> list[str]:
        """Script-split tokens plus their particle-stripped stems, de-duplicated.

        Both forms are kept on purpose: downstream lexical matching against Korean
        column comments wants recall, and a spurious stem costs far less than a miss.
        """
        out: list[str] = []
        seen: set[str] = set()
        for surface, _s, _e, _kind in _scan(_normalize_text(text)):
            for form in (surface, _strip_particle(surface)):
                if form and form not in seen:
                    seen.add(form)
                    out.append(form)
        return out

    # -- main entry point -------------------------------------------------- #

    def normalize(self, text: str) -> NormalizedQuestion:
        norm = _normalize_text(text)
        mask = bytearray(len(norm))

        date_ranges, date_points = self._extract_dates(norm, mask)
        amounts = _extract_amounts(norm, mask)
        comparisons = _extract_comparisons(norm)
        group_by = _extract_group_by(norm)
        top_k = _extract_top_k(norm)
        cues = _detect_cues(norm, has_date=bool(date_ranges or date_points))

        entities: dict[str, Any] = {
            "date_range": date_ranges,
            "date_point": date_points,
            "amount": amounts,
            "comparison": comparisons,
            "group_by_hint": group_by,
            # NOTE: a bool-map, not a list — ``NormalizedQuestion.entities`` is typed
            # ``dict[str, list[Any]]`` but the router consumes cues by name.
            "cues": cues,
        }
        if top_k is not None:
            entities["top_k"] = top_k

        nq = NormalizedQuestion(
            raw=text,
            normalized=norm,
            tokens=self.tokenize(norm),
            entities=entities,
            value_candidates=_extract_value_candidates(norm, mask),
            intent=_classify_intent(norm, cues, top_k),
        )
        log.debug(
            "question normalized",
            intent=nq.intent,
            tokens=len(nq.tokens),
            dates=len(date_ranges),
            amounts=len(amounts),
        )
        return nq

    # -- date extraction --------------------------------------------------- #

    def _extract_dates(
        self, text: str, mask: bytearray
    ) -> tuple[list[tuple[str, tuple[str, str]]], list[tuple[str, str]]]:
        ranges: list[tuple[str, tuple[str, str]]] = []
        points: list[tuple[str, str]] = []
        for pattern, handler in self._date_rules:
            for m in pattern.finditer(text):
                if not _free(mask, m.start(), m.end()):
                    continue
                resolved = handler(m)
                if resolved is None:
                    continue
                _mark(mask, m.start(), m.end())
                surface = m.group(0).strip()
                start, end = resolved
                if start == end:
                    points.append((surface, start))
                ranges.append((surface, (start, end)))
        # Left-to-right reading order keeps "A부터 B까지" style questions sane.
        ranges.sort(key=lambda item: text.find(item[0]))
        points.sort(key=lambda item: text.find(item[0]))
        return ranges, points

    # -- handlers (each returns an inclusive YYYYMMDD range, or None) ------- #

    def _r_ymd_kor(self, m: re.Match[str]) -> _Resolved | None:
        return _point(_safe_date(int(m[1]), int(m[2]), int(m[3])))

    def _r_ymd_sep(self, m: re.Match[str]) -> _Resolved | None:
        return _point(_safe_date(int(m[1]), int(m[2]), int(m[3])))

    def _r_ymd_flat(self, m: re.Match[str]) -> _Resolved | None:
        raw = m[1]
        return _point(_safe_date(int(raw[:4]), int(raw[4:6]), int(raw[6:])))

    def _r_y_qtr(self, m: re.Match[str]) -> _Resolved | None:
        return _quarter_range(int(m[1]), int(m[2]))

    def _r_y_half(self, m: re.Match[str]) -> _Resolved | None:
        return _half_range(int(m[1]), m[2])

    def _r_ym_kor(self, m: re.Match[str]) -> _Resolved | None:
        return _checked_month(int(m[1]), int(m[2]))

    def _r_ym_sep(self, m: re.Match[str]) -> _Resolved | None:
        return _checked_month(int(m[1]), int(m[2]))

    def _r_rel_qtr(self, m: re.Match[str]) -> _Resolved | None:
        return _quarter_range(self.today.year + _REL_YEAR_OFFSET[m[1]], int(m[2]))

    def _r_rel_half(self, m: re.Match[str]) -> _Resolved | None:
        return _half_range(self.today.year + _REL_YEAR_OFFSET[m[1]], m[2])

    def _r_rel_month(self, m: re.Match[str]) -> _Resolved | None:
        return _checked_month(self.today.year + _REL_YEAR_OFFSET[m[1]], int(m[2]))

    def _r_year(self, m: re.Match[str]) -> _Resolved | None:
        year = int(m[1])
        return _year_range(year) if 1900 <= year <= 2100 else None

    def _r_recent(self, m: re.Match[str]) -> _Resolved | None:
        """``최근 N개월`` is a closed window ending today, so the start is inclusive."""
        n, unit = int(m[1]), m[2]
        if n <= 0:
            return None
        if unit == "일":
            start = self.today - timedelta(days=n - 1)
        elif unit in ("주", "주일"):
            start = self.today - timedelta(days=7 * n - 1)
        elif unit in ("개월", "달"):
            start = _add_months(self.today, -n) + timedelta(days=1)
        elif unit == "분기":
            start = _add_months(self.today, -3 * n) + timedelta(days=1)
        else:  # 년
            start = _add_months(self.today, -12 * n) + timedelta(days=1)
        return _fmt(start), _fmt(self.today)

    def _r_ytd(self, _m: re.Match[str]) -> _Resolved | None:
        return f"{self.today.year}0101", _fmt(self.today)

    def _r_this_week(self, _m: re.Match[str]) -> _Resolved | None:
        return _week_range(self.today)

    def _r_last_week(self, _m: re.Match[str]) -> _Resolved | None:
        return _week_range(self.today - timedelta(days=7))

    def _r_this_month(self, _m: re.Match[str]) -> _Resolved | None:
        return _month_range(self.today.year, self.today.month)

    def _r_last_month(self, _m: re.Match[str]) -> _Resolved | None:
        prev = _add_months(self.today.replace(day=1), -1)
        return _month_range(prev.year, prev.month)

    def _r_last_qtr(self, _m: re.Match[str]) -> _Resolved | None:
        quarter = (self.today.month - 1) // 3 + 1
        return _quarter_range(self.today.year - 1, 4) if quarter == 1 else _quarter_range(
            self.today.year, quarter - 1
        )

    def _r_this_qtr(self, _m: re.Match[str]) -> _Resolved | None:
        return _quarter_range(self.today.year, (self.today.month - 1) // 3 + 1)

    def _r_qtr(self, m: re.Match[str]) -> _Resolved | None:
        return _quarter_range(self.today.year, int(m[1]))

    def _r_half(self, m: re.Match[str]) -> _Resolved | None:
        return _half_range(self.today.year, m[1])

    def _r_rel_year(self, m: re.Match[str]) -> _Resolved | None:
        return _year_range(self.today.year + _REL_YEAR_OFFSET[m[1]])

    def _r_today(self, _m: re.Match[str]) -> _Resolved | None:
        return _point(self.today)

    def _r_yesterday(self, _m: re.Match[str]) -> _Resolved | None:
        return _point(self.today - timedelta(days=1))

    def _r_day_before(self, _m: re.Match[str]) -> _Resolved | None:
        return _point(self.today - timedelta(days=2))

    def _r_month(self, m: re.Match[str]) -> _Resolved | None:
        return _checked_month(self.today.year, int(m[1]))


_Resolved = tuple[str, str]


def _coerce_today(today: date | str | None) -> date:
    if today is None:
        return date.today()
    if isinstance(today, date):
        return today
    digits = today.replace("-", "").replace(".", "").replace("/", "").strip()
    if len(digits) != 8 or not digits.isdigit():
        raise ValueError(f"today must be a date or YYYYMMDD string, got {today!r}")
    return date(int(digits[:4]), int(digits[4:6]), int(digits[6:]))


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _point(day: date | None) -> _Resolved | None:
    return None if day is None else (_fmt(day), _fmt(day))


def _checked_month(year: int, month: int) -> _Resolved | None:
    if not (1 <= month <= 12 and 1900 <= year <= 2100):
        return None
    return _month_range(year, month)


def _scan(text: str) -> list[tuple[str, int, int, str]]:
    return [
        (m.group(0), m.start(), m.end(), m.lastgroup or "sym") for m in _SCAN_RE.finditer(text)
    ]


def _extract_amounts(text: str, mask: bytearray) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for m in _AMOUNT_RE.finditer(text):
        surface = m.group(0).strip()
        if not surface or not _free(mask, m.start(), m.start() + len(surface)):
            continue
        if not _amount_is_monetary(surface, text, m.end()):
            continue
        value = parse_korean_number(surface)
        if value is None or value <= 0:
            continue
        _mark(mask, m.start(), m.start() + len(surface))
        out.append((surface, value))
    return out


def _extract_comparisons(text: str) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    spans: list[tuple[int, int]] = []
    for m in _P_BETWEEN.finditer(text):
        low, high = parse_korean_number(m.group("a")), parse_korean_number(m.group("b"))
        if low is None or high is None:
            continue
        spans.append((m.start(), m.end()))
        out.append(("between", (min(low, high), max(low, high))))
    for pattern in (_P_CMP_SUFFIX, _P_CMP_BODA):
        for m in pattern.finditer(text):
            if any(s <= m.start() < e for s, e in spans):
                continue
            value = parse_korean_number(m.group("val"))
            if value is None:
                continue
            out.append((_CMP_OPS[m.group("op")], value))
    return out


def _extract_group_by(text: str) -> list[str]:
    out: list[str] = []
    for m in _P_GROUPBY.finditer(text):
        stem = m[1]
        following = text[m.end() : m.end() + 1]
        if stem in _BYEOL_EXCLUDE or following in ("도", "다"):
            continue
        dimension = f"{stem}별" if stem in _BYEOL_KEEP_WHOLE else stem
        if dimension not in out:
            out.append(dimension)
    return out


def _extract_top_k(text: str) -> int | None:
    for pattern in _P_TOP_K:
        m = pattern.search(text)
        if m:
            k = int(m[1])
            if 0 < k <= 10000:
                return k
    return None


def _detect_cues(text: str, has_date: bool) -> dict[str, bool]:
    cues = {name: bool(_CUE_RE[name].search(text)) for name in CUE_NAMES}
    cues["temporal"] = cues["temporal"] or has_date
    return cues


def _extract_value_candidates(text: str, mask: bytearray) -> list[str]:
    """Literals worth matching against profiled column values downstream."""
    out: list[str] = []
    seen: set[str] = set()

    def push(value: str) -> None:
        value = value.strip()
        if value and value not in seen:
            seen.add(value)
            out.append(value)

    for m in _P_QUOTED.finditer(text):
        push(m[1])

    for surface, start, end, kind in _scan(text):
        if not _free(mask, start, end):
            continue
        if kind == "lat":
            if len(surface) >= 2 and surface.lower() not in _LATIN_STOPWORDS:
                push(surface)
            continue
        if kind != "han":
            continue
        candidate = _strip_particle(surface) or surface
        if len(candidate) < 2 or candidate.endswith("별"):
            continue
        if candidate in _STOPWORDS or surface in _STOPWORDS:
            continue
        # Analytic tails (``이상인``, ``초과하``, ``얼마인가요``) all begin with a
        # two-syllable stopword; domain nouns (``계약자``, ``월납보험료``) do not.
        if len(candidate) > 2 and candidate[:2] in _STOPWORDS:
            continue
        push(candidate)
    return out[:24]


def _classify_intent(text: str, cues: dict[str, bool], top_k: int | None) -> str:
    if top_k is not None and not cues["ratio"]:
        return "rank"
    for intent, pattern in _INTENT_RE:
        if pattern.search(text):
            return intent
    return "select"
