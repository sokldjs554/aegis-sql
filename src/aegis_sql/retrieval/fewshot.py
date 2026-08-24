"""Few-shot example selection (DAIL-SQL style: masked similarity + MMR).

Two failure modes destroy the value of in-context examples, and both are easy to
walk into:

1. **Literal domination.**  "월납보험료가 20만원 이상인 계약" and "월납보험료가
   50만원 이상인 계약" are the same query; "지점별 계약 건수" and "지점별 청구
   건수" are not.  Similarity computed on the raw surface form fixates on the
   numbers.  So matching runs on the *masked* question — numbers, dates and
   quoted values replaced by placeholders — which is the DAIL-SQL finding that
   question-masked retrieval beats raw-question retrieval.
2. **Redundancy.**  The nearest ``k`` neighbours of a GROUP BY question are
   usually ``k`` near-copies of one GROUP BY, and the model learns nothing about
   the join or the window function it actually needs.  Selection is therefore
   MMR (Carbonell & Goldstein, 1998) with the redundancy term computed over SQL
   *skeletons* — keywords kept, identifiers and literals erased — so diversity
   is measured in query structure rather than in wording.

``diversity`` is the MMR λ: 0.0 is pure relevance, 1.0 is pure novelty; 0.4 is
the default because a Korean insurance workload has a small number of recurring
shapes (rate/ratio, period comparison, ranking, code-joined breakdown) and the
prompt should show several of them.

The masking and skeleton helpers here are deliberately private: the canonical
skeleton utility used by the evaluation harness lives elsewhere, and a retriever
must not drift when that one changes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from aegis_sql.observability.logging import get_logger
from aegis_sql.retrieval.embedder import Embedder
from aegis_sql.types import FewShotExample, NormalizedQuestion

log = get_logger("retrieval.fewshot")

_PLACEHOLDER_DATE = "<DATE>"
_PLACEHOLDER_NUM = "<NUM>"
_PLACEHOLDER_VAL = "<VAL>"

#: Applied in order; earlier patterns win, so dates never decay into numbers.
_MASK_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"'[^']*'|\"[^\"]*\"|“[^”]*”"), _PLACEHOLDER_VAL),
    (re.compile(r"\d{4}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*\d{1,2}"), _PLACEHOLDER_DATE),
    (re.compile(r"\d{4}년\s*\d{1,2}월(\s*\d{1,2}일)?"), _PLACEHOLDER_DATE),
    (re.compile(r"\b\d{8}\b"), _PLACEHOLDER_DATE),
    (re.compile(r"\d{4}년|\d{1,2}월|\d{1,2}분기|\d{1,2}일"), _PLACEHOLDER_DATE),
    (re.compile(r"\d[\d,.]*\s*(억원|만원|천원|억|만|원|퍼센트|%)"), _PLACEHOLDER_NUM),
    (re.compile(r"\d[\d,.]*\s*(건|명|개|위|회|점|세|년차|개월)"), _PLACEHOLDER_NUM),
    (re.compile(r"\d[\d,.]*"), _PLACEHOLDER_NUM),
)

_SQL_TOKEN_RE = re.compile(
    r"'[^']*'|\"[^\"]*\"|[A-Za-z_][A-Za-z0-9_$]*|\d+\.\d+|\d+|<=|>=|<>|!=|\|\||[(),;.*=<>+\-/%]"
)
_SQL_COMMENT_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)

#: Structure-bearing tokens kept verbatim in a skeleton.
_SQL_KEYWORDS = frozenset(
    ["SELECT", "DISTINCT", "FROM", "WHERE", "GROUP", "BY", "ORDER", "HAVING", "LIMIT", "OFFSET", "AS", "ON", "USING", "JOIN", "INNER", "LEFT", "RIGHT", "FULL", "OUTER", "CROSS", "NATURAL", "UNION", "ALL", "INTERSECT", "EXCEPT", "AND", "OR", "NOT", "IN", "EXISTS", "BETWEEN", "LIKE", "GLOB", "IS", "NULL", "CASE", "WHEN", "THEN", "ELSE", "END", "ASC", "DESC", "WITH", "RECURSIVE", "OVER", "PARTITION", "WINDOW", "FILTER", "COUNT", "SUM", "AVG", "MIN", "MAX", "ROUND", "ABS", "CAST", "COALESCE", "NULLIF", "SUBSTR", "SUBSTRING", "STRFTIME", "JULIANDAY", "DATE", "ROW_NUMBER", "RANK", "DENSE_RANK", "NTILE", "LAG", "LEAD"]
)


class FewShotSelector:
    """Retrieves structurally diverse in-context examples for one question."""

    __slots__ = ("examples", "embedder", "_question_matrix", "_skeleton_matrix")

    def __init__(self, examples: list[FewShotExample], embedder: Embedder) -> None:
        self.examples = [_ensure_derived(e) for e in examples]
        self.embedder = embedder
        self._question_matrix: np.ndarray | None = None
        self._skeleton_matrix: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.examples)

    @classmethod
    def from_jsonl(cls, path: str | Path, embedder: Embedder) -> FewShotSelector:
        """Load ``{"question": ..., "sql": ...}`` lines; missing files yield an empty pool."""
        file = Path(path)
        if not file.exists():
            log.warning("few-shot pool not found; prompts will run zero-shot", path=str(file))
            return cls([], embedder)
        examples: list[FewShotExample] = []
        for lineno, line in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                log.warning("skipping malformed few-shot line", path=file.name, line=lineno, error=str(exc))
                continue
            if not payload.get("question") or not payload.get("sql"):
                continue
            examples.append(
                FewShotExample(
                    question=str(payload["question"]),
                    sql=str(payload["sql"]),
                    difficulty=str(payload.get("difficulty", "medium")),
                    schema_name=str(payload.get("schema_name", "default")),
                    masked_question=str(payload.get("masked_question", "")),
                    sql_skeleton=str(payload.get("sql_skeleton", "")),
                    source=str(payload.get("source", "curated")),
                )
            )
        log.info("few-shot pool loaded", examples=len(examples), path=file.name)
        return cls(examples, embedder)

    # -- selection ---------------------------------------------------------- #

    def _ensure_index(self) -> None:
        if self._question_matrix is not None or not self.examples:
            return
        masked = [e.masked_question for e in self.examples]
        skeletons = [e.sql_skeleton for e in self.examples]
        # Fit IDF on a copy: the same embedder instance is shared with schema
        # linking, and mutating it there would invalidate that index.
        fitted = getattr(self.embedder, "fitted", None)
        if callable(fitted):
            self.embedder = fitted(masked + skeletons)
        self._question_matrix = self.embedder.encode(masked)
        self._skeleton_matrix = self.embedder.encode(skeletons)

    def select(self, nq: NormalizedQuestion, k: int, diversity: float = 0.4) -> list[FewShotExample]:
        """Top-``k`` examples by masked-question similarity, re-ranked by MMR."""
        if not self.examples or k <= 0:
            return []
        self._ensure_index()
        assert self._question_matrix is not None and self._skeleton_matrix is not None

        query = _mask_question(nq.normalized or nq.raw)
        relevance = self._question_matrix @ self.embedder.encode([query])[0]
        lam = min(1.0, max(0.0, float(diversity)))

        chosen: list[int] = []
        redundancy = np.zeros(len(self.examples), dtype=np.float32)
        available = np.ones(len(self.examples), dtype=bool)
        for _ in range(min(k, len(self.examples))):
            mmr = (1.0 - lam) * relevance - lam * redundancy
            mmr = np.where(available, mmr, -np.inf)
            pick = int(np.argmax(mmr))
            if not np.isfinite(mmr[pick]):
                break
            chosen.append(pick)
            available[pick] = False
            sims = self._skeleton_matrix @ self._skeleton_matrix[pick]
            redundancy = np.maximum(redundancy, sims)
        return [self.examples[i] for i in chosen]


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #


def _ensure_derived(example: FewShotExample) -> FewShotExample:
    """Fill ``masked_question`` / ``sql_skeleton`` when the pool did not ship them."""
    if not example.masked_question:
        example.masked_question = _mask_question(example.question)
    if not example.sql_skeleton:
        example.sql_skeleton = _sql_skeleton(example.sql)
    return example


def _mask_question(question: str) -> str:
    """Replace literals with placeholders so similarity sees intent, not values."""
    masked = question.strip()
    for pattern, placeholder in _MASK_PATTERNS:
        masked = pattern.sub(placeholder, masked)
    return re.sub(r"\s+", " ", masked).strip()


def _sql_skeleton(sql: str) -> str:
    """Keep the query's shape: keywords survive, identifiers and literals do not."""
    body = _SQL_COMMENT_RE.sub(" ", sql)
    out: list[str] = []
    for token in _SQL_TOKEN_RE.findall(body):
        first = token[0]
        if first in {"'", '"', "“"}:
            out.append("<STR>")
        elif first.isdigit():
            out.append("<NUM>")
        elif first.isalpha() or first == "_":
            upper = token.upper()
            out.append(upper if upper in _SQL_KEYWORDS else "<ID>")
        elif token == ".":
            continue  # qualified names collapse into a single <ID>
        else:
            out.append(token)
    collapsed: list[str] = []
    for token in out:
        if token == "<ID>" and collapsed and collapsed[-1] == "<ID>":
            continue
        collapsed.append(token)
    return " ".join(collapsed)
