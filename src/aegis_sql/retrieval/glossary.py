"""Business glossary lookup — the domain knowledge embeddings cannot supply.

No amount of vector similarity tells a model that 유지율 ("persistency") is
``SUM(CASE WHEN CTRT_STAT_CD IN ('01','05') ...) / COUNT(*)`` over ``TB_CTRT``.
That mapping lives in a company's 용어사전, and injecting it at schema-linking
time is what separates a demo from something an actuary will use.

Two details make this harder than a dictionary lookup:

1. **Korean is agglutinative.**  The question says 실효``된``, 월납보험료``가``,
   지점``별로`` — the term is a prefix, never a standalone token.  Matching
   therefore runs against a whitespace-stripped form of the question, with a
   short-suffix rule that promotes 실효 + 된 back to an exact hit.
2. **Terms nest.**  ``실효`` is a substring of ``실효율``, and a naive substring
   scan reports both.  Matching is greedy longest-first over claimed character
   spans, so the most specific term consumes the text and shorter terms cannot
   fire inside it — the same principle as maximal-munch lexing.

ASCII aliases (``GA``, ``UW``, ``CMS``) are matched with an alnum-boundary
check, because a two-letter substring scan over a Korean question is otherwise
a false-positive generator.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from aegis_sql.observability.logging import get_logger
from aegis_sql.types import GlossaryEntry

log = get_logger("retrieval.glossary")

_WS_RE = re.compile(r"\s+")
_ASCII_RE = re.compile(r"^[a-z0-9_.-]+$")
_TOKEN_SPLIT_RE = re.compile(r"[^0-9a-z가-힣]+")

#: Score tiers.  Exact beats substring beats bag-of-token overlap.
SCORE_EXACT = 1.0
SCORE_SUBSTRING = 0.85
SCORE_OVERLAP_MAX = 0.6
#: Korean particles/endings glued onto a term (실효 + 된, 지점 + 별로).
MAX_AGGLUTINATED_SUFFIX = 2


@dataclass(slots=True)
class GlossaryMatch:
    """A hit with the surface form that produced it — used as linking evidence."""

    entry: GlossaryEntry
    score: float
    surface: str
    kind: str  # "exact" | "substring" | "overlap"


@dataclass(slots=True)
class Glossary:
    """An in-memory 용어사전 with span-aware, longest-match-wins lookup."""

    entries: list[GlossaryEntry] = field(default_factory=list)
    version: int = 1
    domain: str = ""
    _surfaces: list[tuple[str, int]] | None = field(default=None, repr=False, compare=False)

    # -- loading ----------------------------------------------------------- #

    @classmethod
    def load(cls, path: str | Path) -> Glossary:
        """Read ``data/demo/glossary.yaml`` (or any file with the same shape)."""
        file = Path(path)
        if not file.exists():
            log.warning("glossary not found; continuing without business terms", path=str(file))
            return cls()
        payload: dict[str, Any] = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
        entries = [
            GlossaryEntry(
                term=str(raw["term"]),
                aliases=[str(a) for a in (raw.get("aliases") or [])],
                definition=str(raw.get("definition") or ""),
                tables=[str(t) for t in (raw.get("tables") or [])],
                columns=[str(c) for c in (raw.get("columns") or [])],
                sql_hint=raw.get("sql_hint"),
            )
            for raw in (payload.get("entries") or [])
        ]
        glossary = cls(
            entries=entries,
            version=int(payload.get("version", 1)),
            domain=str(payload.get("domain") or ""),
        )
        log.info("glossary loaded", entries=len(entries), path=file.name)
        return glossary

    # -- accessors ---------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.entries)

    def entry(self, term: str) -> GlossaryEntry | None:
        lowered = term.lower()
        return next((e for e in self.entries if e.term.lower() == lowered), None)

    def documents(self) -> list[tuple[str, str]]:
        """``(id, text)`` pairs for indexing the glossary in a vector store."""
        docs: list[tuple[str, str]] = []
        for e in self.entries:
            parts = [e.term]
            if e.aliases:
                parts.append(" ".join(e.aliases))
            if e.definition:
                parts.append(e.definition)
            if e.columns:
                parts.append(" ".join(e.columns))
            docs.append((f"glossary:{e.term}", " | ".join(parts)))
        return docs

    # -- matching ----------------------------------------------------------- #

    def _surface_index(self) -> list[tuple[str, int]]:
        """All term/alias surfaces, longest first — the maximal-munch ordering."""
        if self._surfaces is None:
            surfaces: dict[str, int] = {}
            for idx, entry in enumerate(self.entries):
                for raw in (entry.term, *entry.aliases):
                    key = _squish(raw)
                    # First writer wins: a surface owned by two entries stays with
                    # the one declared first, keeping lookup deterministic.
                    if key and key not in surfaces:
                        surfaces[key] = idx
            self._surfaces = sorted(surfaces.items(), key=lambda kv: (-len(kv[0]), kv[0]))
        return self._surfaces

    def match(self, tokens: Iterable[str], text: str) -> list[GlossaryMatch]:
        """Full-detail lookup; :meth:`lookup` is the ``(entry, score)`` view of it."""
        token_set = {t.lower() for t in tokens if t}
        squished = _squish(text)
        claimed = [False] * len(squished)
        best: dict[int, GlossaryMatch] = {}

        for surface, idx in self._surface_index():
            span = _find_unclaimed(squished, surface, claimed)
            if span is None:
                continue
            start, end = span
            for i in range(start, end):
                claimed[i] = True
            exact = surface in token_set or _is_agglutinated(surface, token_set)
            match = GlossaryMatch(
                entry=self.entries[idx],
                score=SCORE_EXACT if exact else SCORE_SUBSTRING,
                surface=surface,
                kind="exact" if exact else "substring",
            )
            _keep_best(best, idx, match)

        # Tier 3: multi-token terms that never appear verbatim (e.g. "loss ratio"
        # against a question that only says "ratio").
        for idx, entry in enumerate(self.entries):
            if idx in best:
                continue
            overlap = _best_overlap(entry, token_set)
            if overlap is None:
                continue
            ratio, surface = overlap
            _keep_best(
                best,
                idx,
                GlossaryMatch(entry, round(SCORE_OVERLAP_MAX * ratio, 4), surface, "overlap"),
            )

        return sorted(best.values(), key=lambda m: (-m.score, -len(m.surface), m.entry.term))

    def lookup(self, tokens: list[str], text: str) -> list[tuple[GlossaryEntry, float]]:
        """Rank glossary entries against a question.  Deduped by term, best first."""
        return [(m.entry, m.score) for m in self.match(tokens, text)]


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #


def _squish(text: str) -> str:
    """Lowercase and drop all whitespace — Korean compounds are written both ways."""
    return _WS_RE.sub("", text.lower())


def _find_unclaimed(haystack: str, needle: str, claimed: list[bool]) -> tuple[int, int] | None:
    """First occurrence of ``needle`` whose characters are all still unclaimed."""
    if not needle:
        return None
    ascii_only = bool(_ASCII_RE.match(needle))
    start = 0
    while True:
        pos = haystack.find(needle, start)
        if pos < 0:
            return None
        end = pos + len(needle)
        if ascii_only and not _has_ascii_boundary(haystack, pos, end):
            start = pos + 1
            continue
        if not any(claimed[pos:end]):
            return pos, end
        start = pos + 1


def _has_ascii_boundary(haystack: str, start: int, end: int) -> bool:
    before = haystack[start - 1] if start > 0 else ""
    after = haystack[end] if end < len(haystack) else ""
    return not (before.isascii() and before.isalnum()) and not (after.isascii() and after.isalnum())


def _is_agglutinated(surface: str, token_set: set[str]) -> bool:
    """True when a question token is ``surface`` plus a short Korean ending."""
    for token in token_set:
        if not token.startswith(surface) or token == surface:
            continue
        suffix = token[len(surface) :]
        if len(suffix) <= MAX_AGGLUTINATED_SUFFIX and all("가" <= ch <= "힣" for ch in suffix):
            return True
    return False


def _best_overlap(entry: GlossaryEntry, token_set: set[str]) -> tuple[float, str] | None:
    best: tuple[float, str] | None = None
    for raw in (entry.term, *entry.aliases):
        parts = [p for p in _TOKEN_SPLIT_RE.split(raw.lower()) if p]
        if len(parts) < 2:
            continue
        ratio = sum(1 for p in parts if p in token_set) / len(parts)
        if ratio >= 0.5 and (best is None or ratio > best[0]):
            best = (ratio, raw)
    return best


def _keep_best(store: dict[int, GlossaryMatch], idx: int, candidate: GlossaryMatch) -> None:
    current = store.get(idx)
    if current is None or (candidate.score, len(candidate.surface)) > (current.score, len(current.surface)):
        store[idx] = candidate
