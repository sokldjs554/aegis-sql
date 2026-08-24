"""Korean question augmentation — the variance the back-translator cannot produce.

Back-translation gives the corpus one *correct* Korean sentence per SQL program.
Real users do not send correct sentences.  They send 반말 and 존댓말 in the same
session, drop 조사, mistype 중성 on a phone keyboard, glue words together or split
them apart, and say 월보험료 when the 용어사전 says 월납보험료.  A model trained on
the back-translated corpus alone learns the *template*, not the language: it is
brittle in exactly the places where a deployed engine is judged.

So the point of this module is not to inflate the row count.  It is to widen the
input distribution along the axes that actually vary in production, each as an
independent, composable transform that returns ``None`` when it does not apply:

* **Lexical** — 용어사전 aliases (유지율 ↔ 보유율) and everyday contractions.
* **Morphological** — 조사 교체/생략, honorific register, sentence-final politeness.
* **Syntactic** — adverbial phrases relocated, which Korean permits freely.
* **Surface noise** — 띄어쓰기 errors and jamo-level typos.

The typo transform is the reason this file implements Hangul composition itself.
A character-level substitution ("가" → "각") is not the mistake Korean typists
make; the real one is at the *jamo* level — a 중성 slipping to its keyboard
neighbour (ㅏ→ㅑ) or a 종성 appearing or vanishing.  Reproducing that needs the
standard ``0xAC00 + ((초 × 21) + 중) × 28 + 종`` arithmetic, ~15 lines and no
dependency, which is why no morphological analyser is imported here.

The invariant
-------------
**Augmentation may change how a question is worded; it may never change what it
asks for.**  Every candidate is gated on a *literal signature* — the multiset of
number-with-unit spans (``2024년``, ``20만원``, ``10건``) extracted before and
after — and any candidate that perturbs one is discarded rather than repaired.
Without that gate, one typo inside "20만원" silently produces a training pair
whose question and SQL disagree, and execution-based verification cannot catch
it: the SQL still runs and still returns rows.  Transforms additionally refuse to
edit inside a protected span, so rejections stay rare rather than load-bearing.

Everything is deterministic in ``(seed, question)``: the RNG is derived from both,
so a question augmented in one run is augmented identically in the next, whatever
order the corpus is processed in.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from typing import Any

from aegis_sql.observability.logging import get_logger
from aegis_sql.types import GlossaryEntry

log = get_logger("flywheel.augment")

__all__ = [
    "KoreanAugmenter",
    "compose_syllable",
    "decompose_syllable",
    "josa",
    "literal_signature",
    "particle",
    "TRANSFORMS",
]

# --------------------------------------------------------------------------- #
# Hangul jamo arithmetic
# --------------------------------------------------------------------------- #

HANGUL_BASE = 0xAC00
HANGUL_LAST = 0xD7A3

CHOSEONG: tuple[str, ...] = (
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
)
JUNGSEONG: tuple[str, ...] = (
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ",
    "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
)
JONGSEONG: tuple[str, ...] = (
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ",
    "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
)

_JUNG_INDEX = {j: i for i, j in enumerate(JUNGSEONG)}
_JONG_INDEX = {j: i for i, j in enumerate(JONGSEONG)}


def decompose_syllable(char: str) -> tuple[str, str, str] | None:
    """Split a precomposed Hangul syllable into ``(초성, 중성, 종성)``.

    Returns ``None`` for anything outside the 가–힣 block (Latin, digits,
    punctuation, standalone jamo), which is what makes call sites total.
    """
    code = ord(char)
    if not HANGUL_BASE <= code <= HANGUL_LAST:
        return None
    offset = code - HANGUL_BASE
    return CHOSEONG[offset // 588], JUNGSEONG[(offset % 588) // 28], JONGSEONG[offset % 28]


def compose_syllable(cho: str, jung: str, jong: str = "") -> str:
    """Inverse of :func:`decompose_syllable`; raises on an unknown jamo."""
    return chr(HANGUL_BASE + (CHOSEONG.index(cho) * 21 + _JUNG_INDEX[jung]) * 28 + _JONG_INDEX[jong])


def has_batchim(word: str) -> bool:
    """True when the last syllable carries a final consonant (받침)."""
    for char in reversed(word):
        parts = decompose_syllable(char)
        if parts is not None:
            return bool(parts[2])
        if char.isdigit():
            # Korean readings of digits: 0(영) 1(일) 3(삼) 6(육) 7(칠) 8(팔) end in a
            # consonant, the rest do not.
            return char in "01368"
    return False


def particle(word: str, pair: str) -> str:
    """The correct allomorph on its own: ``particle("계약", "은/는") -> "은"``."""
    with_batchim, _, without = pair.partition("/")
    return with_batchim if has_batchim(word) else without


def josa(word: str, pair: str) -> str:
    """``word`` with the right particle attached: ``josa("계약", "은/는") -> "계약은"``."""
    return word + particle(word, pair)


# --------------------------------------------------------------------------- #
# the invariant: numbers, dates and code values must survive untouched
# --------------------------------------------------------------------------- #

#: A number together with the unit glued to it.  Matching the unit as well is the
#: whole point: a typo inside "만원" changes the amount by four orders of
#: magnitude while leaving every digit in place.
_LITERAL_RE = re.compile(
    r"\d[\d,]*(?:\.\d+)?(?:\s*(?:억|천만|백만|만|천|원|건|개|명|년|월|일|분기|반기|회|위|점|%|퍼센트|배))*"
)
_SPACE_RE = re.compile(r"\s+")
_HANGUL_RE = re.compile(r"[가-힣]")


def literal_signature(text: str) -> Counter[str]:
    """Whitespace-insensitive multiset of the value spans a question commits to."""
    return Counter(_SPACE_RE.sub("", m.group(0)) for m in _LITERAL_RE.finditer(text))


def _protected_positions(text: str) -> set[int]:
    out: set[int] = set()
    for match in _LITERAL_RE.finditer(text):
        out.update(range(match.start(), match.end()))
    return out


# --------------------------------------------------------------------------- #
# transform tables
# --------------------------------------------------------------------------- #

#: Longest alternatives first so 에서 is not truncated to 에.
_PARTICLE_RE = re.compile(
    r"(?<=[가-힣])(별로|에서|으로|부터|까지|은|는|이|가|을|를|와|과|의|에|로|별)(?=[\s,.?!]|$)"
)
#: ``particle -> the wrong-but-real allomorph a hurried typist writes``.
_PARTICLE_SWAP: dict[str, tuple[str, ...]] = {
    "은": ("는",), "는": ("은",), "이": ("가",), "가": ("이",),
    "을": ("를",), "를": ("을",), "와": ("과",), "과": ("와",),
    "으로": ("로",), "로": ("으로",), "별로": ("별", "마다"), "별": ("별로", "마다"),
    "에서": ("서",), "의": ("",),
}
#: Particles that colloquial Korean simply drops.
_PARTICLE_DROPPABLE = frozenset({"은", "는", "이", "가", "을", "를", "의"})

#: Sentence-final request forms, all semantically identical.
_REQUEST_ENDINGS: tuple[str, ...] = (
    "알려줘", "알려주세요", "조회해줘", "조회해주세요", "뽑아줘", "뽑아주세요",
    "보여줘", "보여주세요", "정리해줘", "집계해줘", "확인해줘",
)
#: Interrogative realisations — same request, no verb.
_QUESTION_ENDINGS: tuple[str, ...] = ("가 궁금해요", "가 어떻게 되나요", "는 얼마나 되나요", "는?")
_ENDING_RE = re.compile("(" + "|".join(sorted(_REQUEST_ENDINGS, key=len, reverse=True)) + r")\s*[.?!]?\s*$")
_QUESTION_TAIL_RE = re.compile(r"(가 궁금해요|가 어떻게 되나요|는 얼마나 되나요|는)\s*[.?!]?\s*$")

_HONORIFIC_SHIFTS: tuple[tuple[str, str], ...] = (
    ("알려줘", "알려주시기 바랍니다"),
    ("알려주세요", "알려주시겠어요"),
    ("조회해줘", "조회 부탁드립니다"),
    ("보여줘", "보여주시겠습니까"),
    ("뽑아줘", "뽑아주시면 감사하겠습니다"),
    ("정리해줘", "정리해 주시기 바랍니다"),
)

#: Particle pairs repaired after a substitution changes the preceding 받침.
_REPAIRABLE_PARTICLES: tuple[tuple[str, str], ...] = (
    ("으로", "로"), ("은", "는"), ("이", "가"), ("을", "를"), ("과", "와"),
)

#: Everyday contractions.  Only meaning-preserving pairs belong here.
_CONTRACTIONS: dict[str, str] = {
    "계약건수": "건수",
    "월납보험료": "월보험료",
    "총가입금액": "가입금액",
    "보험금지급액": "지급액",
    "보험금 지급액": "지급액",
    "청구금액": "청구액",
    "고객등급": "등급",
    "계약상태": "상태",
    "납입방법": "납입",
    "만족도점수": "만족도",
    "이상징후점수": "이상징후",
    "처리완료일자": "완료일",
    "계약체결일자": "체결일",
    "청구접수일자": "접수일",
    "인수심사": "심사",
    "보험설계사": "설계사",
    "설계사별": "FC별",
    "평균값": "평균",
    "합계금액": "합계",
}

#: 두벌식 keyboard neighbours — the vowels people actually swap on a phone.
_VOWEL_NEIGHBOURS: dict[str, tuple[str, ...]] = {
    "ㅏ": ("ㅑ", "ㅓ"), "ㅑ": ("ㅏ",), "ㅓ": ("ㅕ", "ㅏ"), "ㅕ": ("ㅓ",),
    "ㅗ": ("ㅛ", "ㅜ"), "ㅛ": ("ㅗ",), "ㅜ": ("ㅠ", "ㅗ"), "ㅠ": ("ㅜ",),
    "ㅐ": ("ㅔ", "ㅏ"), "ㅔ": ("ㅐ", "ㅓ"), "ㅒ": ("ㅖ",), "ㅖ": ("ㅒ",),
    "ㅡ": ("ㅜ", "ㅣ"), "ㅣ": ("ㅡ", "ㅐ"),
}
#: 종성 confusions, including the very common appear/disappear pair.
_FINAL_NEIGHBOURS: dict[str, tuple[str, ...]] = {
    "": ("ㅇ", "ㄴ"), "ㅇ": ("", "ㄴ"), "ㄴ": ("ㅇ", ""), "ㄱ": ("ㄲ", ""),
    "ㄲ": ("ㄱ",), "ㅅ": ("ㅆ", "ㅈ"), "ㅆ": ("ㅅ",), "ㄹ": ("", "ㄴ"), "ㅁ": ("ㅂ", "ㄴ"),
}

#: Adverbial phrase tails — Korean lets these move freely, other chunks do not.
_ADVERBIAL_TAIL_RE = re.compile(r"(에|에서|부터|까지|간|동안|별로|별|기준으로|이상|이하|초과|미만)$")
_DIGIT_RE = re.compile(r"\d")
#: Words that bind leftward onto a time expression ("최근 6개월간").
_TIME_MODIFIERS = frozenset({"최근", "지난", "작년", "올해", "전년", "전체", "가장"})

_FILLERS: tuple[str, ...] = ("좀", "한번")
#: A filler may only slot in front of a *request verb*; wedging one into
#: "…얼마나 되나요?" is not colloquial, it is broken.
_VERB_TAIL_RE = re.compile(r"(알려|보여|조회|뽑아|집계|정리|확인|주세요|주십시오|바랍니다|해줘)")

#: Public name of every transform, in a fixed order so sampling is reproducible.
TRANSFORMS: tuple[str, ...] = (
    "_synonym_swap",
    "_particle_variation",
    "_politeness_variation",
    "_word_order",
    "_abbreviate",
    "_typo",
    "_spacing",
    "_honorific_noise",
    "_filler",
)


class KoreanAugmenter:
    """Composable, invariant-checked Korean paraphrasing for the flywheel."""

    def __init__(self, glossary: Any = None, seed: int = 20260824) -> None:
        self.seed = int(seed)
        self._surfaces: list[tuple[str, tuple[str, ...]]] = _build_surfaces(glossary)
        log.debug("augmenter ready", glossary_terms=len(self._surfaces), transforms=len(TRANSFORMS))

    # -- public ------------------------------------------------------------ #

    def augment(self, question: str, n: int) -> list[str]:
        """Up to ``n`` distinct rewordings of ``question`` that ask the same thing."""
        return [text for text, _ in self.augment_with_ops(question, n)]

    def augment_with_ops(self, question: str, n: int) -> list[tuple[str, list[str]]]:
        """:meth:`augment` plus the transform chain that produced each variant.

        ``build_dataset`` records the chain as the pair's provenance, so a later
        error analysis can ask "does the model fail on typos or on 조사 생략?"
        instead of only "does it fail on augmented data?".
        """
        source = (question or "").strip()
        if n <= 0 or not source:
            return []
        rng = random.Random(f"{self.seed}:{source}")
        baseline = literal_signature(source)
        seen = {_normalise(source)}
        out: list[tuple[str, list[str]]] = []

        for _ in range(max(16, n * 14)):
            if len(out) >= n:
                break
            chain = rng.sample(list(TRANSFORMS), rng.choice((1, 1, 2, 2, 3)))
            text, applied = source, []
            for name in chain:
                candidate = getattr(self, name)(text, rng)
                if candidate and candidate != text:
                    text, applied = candidate, [*applied, name.lstrip("_")]
            if not applied:
                continue
            text = _normalise(text)
            # The gate, not a repair: a variant that moved a value is thrown away.
            if literal_signature(text) != baseline or text in seen:
                continue
            seen.add(text)
            out.append((text, applied))

        if len(out) < n:
            log.debug("augmentation under-filled", requested=n, produced=len(out))
        return out

    # -- transforms -------------------------------------------------------- #

    def _synonym_swap(self, text: str, rng: random.Random) -> str | None:
        """Swap a 용어사전 term for one of its declared aliases."""
        hits = [(surface, alts) for surface, alts in self._surfaces if surface in text]
        if not hits:
            return None
        surface, alternatives = hits[rng.randrange(len(hits))]
        pool = [a for a in alternatives if a != surface]
        if not pool:
            return None
        replacement = pool[rng.randrange(len(pool))]
        at = text.index(surface)
        swapped = text[:at] + replacement + text[at + len(surface) :]
        return _repair_particle(swapped, at + len(replacement), replacement)

    def _particle_variation(self, text: str, rng: random.Random) -> str | None:
        """Replace a 조사 with a neighbouring allomorph, or drop it entirely."""
        matches = [m for m in _PARTICLE_RE.finditer(text)]
        if not matches:
            return None
        match = matches[rng.randrange(len(matches))]
        particle = match.group(1)
        options = list(_PARTICLE_SWAP.get(particle, ()))
        if particle in _PARTICLE_DROPPABLE:
            options.append("")
        if not options:
            return None
        return text[: match.start(1)] + options[rng.randrange(len(options))] + text[match.end(1) :]

    def _politeness_variation(self, text: str, rng: random.Random) -> str | None:
        """Swap the sentence-final request form (반말 / 존댓말 / 의문형)."""
        match = _ENDING_RE.search(text)
        if match:
            pool = [e for e in (*_REQUEST_ENDINGS, *_QUESTION_ENDINGS) if e != match.group(1)]
            choice = pool[rng.randrange(len(pool))]
            body = text[: match.start(1)].rstrip()
            if choice in _QUESTION_ENDINGS:
                body = _strip_object_particle(body)
                return f"{body}{choice}?" if choice.endswith("는") else f"{body}{choice}"
            return f"{body} {choice}." if body and not body.endswith(" ") else f"{body}{choice}."
        tail = _QUESTION_TAIL_RE.search(text)
        if tail:
            body = text[: tail.start(1)].rstrip()
            choice = _REQUEST_ENDINGS[rng.randrange(len(_REQUEST_ENDINGS))]
            return f"{body}{particle(body, '을/를')} {choice}." if body else None
        return None

    def _word_order(self, text: str, rng: random.Random) -> str | None:
        """Relocate a whole adverbial phrase — the one movement Korean always allows.

        The unit that moves is the *span*, not the chunk: "2024년 3분기에" is one
        adverbial and splitting it would scatter a date across the sentence while
        leaving the literal signature intact, which is precisely the corruption
        the invariant gate cannot see.
        """
        chunks = text.split(" ")
        if len(chunks) < 4:
            return None
        spans: list[tuple[int, int]] = []
        for end, chunk in enumerate(chunks[:-1]):
            if not (_ADVERBIAL_TAIL_RE.search(chunk) and _HANGUL_RE.search(chunk)):
                continue
            start = end
            while start > 0 and (_DIGIT_RE.search(chunks[start - 1]) or chunks[start - 1] in _TIME_MODIFIERS):
                start -= 1
            if len(chunks) - (end - start + 1) >= 2:
                spans.append((start, end))
        if not spans:
            return None
        start, end = spans[rng.randrange(len(spans))]
        span = chunks[start : end + 1]
        rest = chunks[:start] + chunks[end + 1 :]
        target = len(rest) - 1 if start == 0 else 0
        return " ".join(rest[:target] + span + rest[target:])

    def _abbreviate(self, text: str, rng: random.Random) -> str | None:
        """Contract a compound the way an analyst types it in a hurry."""
        hits = [(long, short) for long, short in _CONTRACTIONS.items() if long in text]
        if not hits:
            return None
        long, short = hits[rng.randrange(len(hits))]
        at = text.index(long)
        contracted = text[:at] + short + text[at + len(long) :]
        return _repair_particle(contracted, at + len(short), short)

    def _typo(self, text: str, rng: random.Random) -> str | None:
        """Perturb one syllable's 중성 or 종성 — a real keyboard slip, not a random char."""
        protected = _protected_positions(text)
        positions = [
            i
            for i, char in enumerate(text)
            if i not in protected and decompose_syllable(char) is not None
        ]
        if not positions:
            return None
        for _ in range(6):
            index = positions[rng.randrange(len(positions))]
            cho, jung, jong = decompose_syllable(text[index])  # type: ignore[misc]
            if rng.random() < 0.6:
                pool = _VOWEL_NEIGHBOURS.get(jung, ())
                if pool:
                    jung = pool[rng.randrange(len(pool))]
                    return text[:index] + compose_syllable(cho, jung, jong) + text[index + 1 :]
            pool = _FINAL_NEIGHBOURS.get(jong, ())
            if pool:
                jong = pool[rng.randrange(len(pool))]
                return text[:index] + compose_syllable(cho, jung, jong) + text[index + 1 :]
        return None

    def _spacing(self, text: str, rng: random.Random) -> str | None:
        """Inject or remove a 띄어쓰기 error between Hangul syllables."""
        protected = _protected_positions(text)
        joins = [
            i
            for i, char in enumerate(text)
            if char == " "
            and 0 < i < len(text) - 1
            and _HANGUL_RE.match(text[i - 1])
            and _HANGUL_RE.match(text[i + 1])
            and (i - 1) not in protected
            and (i + 1) not in protected
        ]
        splits = [
            i
            for i in range(1, len(text))
            if _HANGUL_RE.match(text[i])
            and _HANGUL_RE.match(text[i - 1])
            and i not in protected
            and (i - 1) not in protected
        ]
        actions = [("join", joins)] if joins else []
        if splits:
            actions.append(("split", splits))
        if not actions:
            return None
        action, positions = actions[rng.randrange(len(actions))]
        index = positions[rng.randrange(len(positions))]
        if action == "join":
            return text[:index] + text[index + 1 :]
        return text[:index] + " " + text[index:]

    def _honorific_noise(self, text: str, rng: random.Random) -> str | None:
        """Shift the honorific register without touching the request itself."""
        hits = [(src, dst) for src, dst in _HONORIFIC_SHIFTS if src in text]
        if not hits:
            return None
        src, dst = hits[rng.randrange(len(hits))]
        return text.replace(src, dst, 1)

    def _filler(self, text: str, rng: random.Random) -> str | None:
        """Insert a discourse filler (혹시 / 좀 / 한번)."""
        chunks = text.split(" ")
        options: list[str] = ["혹시"] if "혹시" not in text else []
        if len(chunks) >= 2 and _VERB_TAIL_RE.search(chunks[-1]):
            options.extend(f for f in _FILLERS if f not in text)
        if not options:
            return None
        filler = options[rng.randrange(len(options))]
        if filler == "혹시":
            return f"혹시 {text}"
        return " ".join([*chunks[:-1], filler, chunks[-1]])


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _build_surfaces(glossary: Any) -> list[tuple[str, tuple[str, ...]]]:
    """``[(surface, all interchangeable surfaces), ...]`` from the 용어사전.

    ASCII-only aliases (``persistency``, ``APE``) and digit-bearing ones
    (``13회차유지율``) are dropped: substituting them either switches language
    mid-sentence or violates the literal invariant.
    """
    if glossary is None:
        return []
    entries = getattr(glossary, "entries", glossary)
    out: list[tuple[str, tuple[str, ...]]] = []
    for entry in entries:
        if not isinstance(entry, GlossaryEntry):
            continue
        surfaces = tuple(
            dict.fromkeys(
                s
                for s in (entry.term, *entry.aliases)
                if _HANGUL_RE.search(s) and not any(c.isdigit() for c in s)
            )
        )
        if len(surfaces) < 2:
            continue
        out.extend((surface, surfaces) for surface in surfaces)
    # Longest surface first: 월납보험료 must win over 보험료 when both are present.
    out.sort(key=lambda item: (-len(item[0]), item[0]))
    return out


def _repair_particle(text: str, at: int, word: str) -> str:
    """Re-agree the 조사 at ``at`` with ``word``'s new final consonant.

    Substituting 표준인수 → 표준체승낙 leaves "표준체승낙로", which is not a
    different register — it is simply wrong, and wrong Korean in the *clean*
    part of the corpus is noise the model cannot learn around.
    """
    for with_batchim, without in _REPAIRABLE_PARTICLES:
        for surface in (with_batchim, without):
            end = at + len(surface)
            if not text.startswith(surface, at):
                continue
            if end < len(text) and _HANGUL_RE.match(text[end]):
                continue
            return text[:at] + particle(word, f"{with_batchim}/{without}") + text[end:]
    return text


def _strip_object_particle(text: str) -> str:
    """Drop a trailing 을/를/이/가 so an interrogative ending attaches cleanly."""
    return text[:-1] if text and text[-1] in "을를이가은는" else text


def _normalise(text: str) -> str:
    return _SPACE_RE.sub(" ", text).strip()
