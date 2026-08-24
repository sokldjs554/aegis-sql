"""A deterministic, offline LLM — the reason the frontier tier is testable at all.

Every interesting property of the LLM tier lives *around* the model call: the
prompt registry renders a 6 KB Korean prompt, the completion is fenced prose
that has to be parsed back into one statement, ``n`` samples get voted on by
execution signature, and the whole thing is billed in USD.  None of that needs a
network — but all of it is normally locked behind an API key, which CI does not
have and a reviewer cloning the repo does not have either.

:class:`MockLLM` unlocks it.  It is a real :class:`~aegis_sql.llm.base.LLMClient`
that answers from a fixture table, so ``provider: mock`` runs the identical code
path (prompt → completion → ``extract_sql`` → guard → execute → vote) end to end
with zero cost and zero variance.

Three design choices are what make it useful rather than a stub:

1. **It reads the question out of the rendered prompt, not from a parameter.**
   The only thing it is given is ``list[Message]``, exactly like a real
   provider, and it recovers the question from the ``【질문】`` block.  A prompt
   edit that drops the question therefore makes the mock miss its fixture and a
   test fail — the prompt set is covered by the same assertions as the code.
2. **It answers in the shape the prompt asked for.**  The prompt set contains
   five distinct output contracts (SQL block, Korean summary, JSON verdict, a
   single back-translated question, newline-separated paraphrases), so the mock
   keys off the contract line in the prompt and emits a matching payload.  A
   mock that always returned SQL would make the answer/self-check/flywheel paths
   untestable offline.
3. **``complete_n`` returns semantics-preserving rewrites.**  The variants differ
   only in ``COUNT(*)``/``COUNT(1)``, an explicit projection alias and line
   folding, so execution-signature voting *must* collapse them into a single
   group while naive string voting shatters them into singletons — which is the
   exact behaviour :mod:`aegis_sql.verify.selfconsistency` exists to have.

Token counts come from the same cheap estimator the schema cards are measured
with; cost is 0.0 by the price table, not by a special case.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence

from aegis_sql.llm.base import LLMResponse, Message, estimate_cost
from aegis_sql.observability.logging import get_logger
from aegis_sql.schema.card import token_estimate
from aegis_sql.types import now_ms

log = get_logger("llm.mock")

#: Markers the prompt set uses to introduce the question, in priority order.
_QUESTION_MARKERS = ("【질문】", "【원 질문】")
_BLOCK_MARKER = re.compile(r"^\s*【")
_TABLE_TOKEN = re.compile(r"\bTB_[A-Z0-9_]+\b")
_COUNT_STAR = re.compile(r"COUNT\s*\(\s*\*\s*\)", re.IGNORECASE)
#: A single unaliased aggregate projection — the safe place to add ``AS`` to.
_BARE_AGGREGATE = re.compile(
    r"^(SELECT\s+(?:COUNT|SUM|AVG|MIN|MAX)\s*\([^()]*\))(\s+FROM\b)", re.IGNORECASE
)

#: ``(marker found in the prompt, kind)`` — first match wins.  The markers are the
#: *output contract* lines of ``configs/prompts/default.yaml``, so the mock's
#: answer shape follows the prompt set when it changes.
_KIND_MARKERS: tuple[tuple[str, str], ...] = (
    ("【결과】", "summary"),
    ("JSON", "json"),
    ("질문 한 문장만", "question"),
    ("한 줄에 하나씩", "lines"),
)

_DEFAULT_TABLE = "TB_CTRT"


def extract_question(prompt: str) -> str:
    """Recover the natural-language question from a rendered prompt.

    Handles both layouts the prompt set uses: the marker on its own line with the
    question underneath (``nl2sql.user``) and the marker inline (``repair.user``).
    Returns ``""`` when no marker is present, which is what makes a missing
    question observable instead of silently defaulted.
    """
    for marker in _QUESTION_MARKERS:
        index = prompt.find(marker)
        if index < 0:
            continue
        rest = prompt[index + len(marker) :]
        head, _, tail = rest.partition("\n")
        if head.strip():
            return head.strip()
        lines: list[str] = []
        for line in tail.splitlines():
            if _BLOCK_MARKER.match(line) or (lines and not line.strip()):
                break
            if line.strip():
                lines.append(line.strip())
        if lines:
            return " ".join(lines)
    return ""


class MockLLM:
    """Offline :class:`~aegis_sql.llm.base.LLMClient` backed by fixtures.

    Parameters
    ----------
    fixtures:
        ``{question substring: SQL}``.  The longest matching key wins, so a
        specific fixture can shadow a general one deterministically.
    fallback:
        Called with the question when no fixture matches — the pipeline passes
        the template tier's compile function here, which turns the mock into a
        "frontier model that is exactly as good as the deterministic tier" and
        keeps offline demos answering real questions.
    """

    name = "mock"

    def __init__(
        self,
        fixtures: Mapping[str, str] | None = None,
        fallback: Callable[[str], str | None] | None = None,
        model: str = "mock",
    ) -> None:
        self.fixtures: dict[str, str] = dict(fixtures or {})
        self.fallback = fallback
        self.model = model
        #: Observability for tests: how many completions were served, and the last
        #: prompt seen (asserting on it is how prompt regressions get caught).
        self.calls = 0
        self.last_prompt = ""

    # -- LLMClient --------------------------------------------------------- #

    def available(self) -> bool:
        """Always true.  The mock is the floor the engine never falls through."""
        return True

    def complete(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        **kwargs: object,
    ) -> LLMResponse:
        return self._respond(messages, variant=0)

    def complete_n(
        self,
        messages: list[Message],
        n: int,
        temperature: float | None = None,
        **kwargs: object,
    ) -> list[LLMResponse]:
        """``n`` samples that differ in surface form only (see the module docstring)."""
        return [self._respond(messages, variant=i) for i in range(max(1, int(n)))]

    # -- internals --------------------------------------------------------- #

    def _respond(self, messages: list[Message], variant: int) -> LLMResponse:
        started = now_ms()
        prompt = "\n".join(m.content for m in messages)
        user_prompt = next(
            (m.content for m in reversed(messages) if m.role == "user"), prompt
        )
        question = extract_question(user_prompt)
        self.calls += 1
        self.last_prompt = user_prompt

        kind = _kind_of(user_prompt)
        if kind == "sql":
            text = f"```sql\n{_apply_variant(self._sql_for(question, user_prompt), variant)}\n```"
        elif kind == "summary":
            text = _mock_summary(user_prompt)
        elif kind == "json":
            text = _mock_json(user_prompt)
        elif kind == "question":
            text = f"{question or '해당 조건에 맞는 데이터'}를 조회하는 질문입니다."
        else:
            text = "\n".join(f"{question} (표현 {i + 1})" for i in range(3))

        prompt_tokens = token_estimate(prompt)
        completion_tokens = token_estimate(text)
        return LLMResponse(
            text=text,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=now_ms() - started,
            cost_usd=estimate_cost(self.model, prompt_tokens, completion_tokens),
            finish_reason="stop",
            raw={"provider": "mock", "kind": kind, "variant": variant, "question": question},
        )

    def _sql_for(self, question: str, prompt: str) -> str:
        for key in sorted(self.fixtures, key=len, reverse=True):
            if key and key in question:
                return self.fixtures[key].strip()
        if self.fallback is not None:
            try:
                sql = self.fallback(question)
            except Exception as exc:  # a fixture-less mock must still answer
                log.warning("mock fallback failed", error=str(exc))
                sql = None
            if sql:
                return sql.strip()
        # Last resort: a statement that is *valid against the schema in the prompt*.
        # Returning something unparseable here would exercise the repair loop on a
        # defect the mock invented, which teaches nobody anything.
        match = _TABLE_TOKEN.search(prompt)
        table = match.group(0) if match else _DEFAULT_TABLE
        return f"SELECT COUNT(*) FROM {table}"


def _kind_of(prompt: str) -> str:
    for marker, kind in _KIND_MARKERS:
        if marker in prompt:
            return kind
    return "sql"


def _apply_variant(sql: str, variant: int) -> str:
    """Semantics-preserving rewrites selected by the bits of ``variant``."""
    out = sql
    if variant & 1:
        out = _COUNT_STAR.sub("COUNT(1)", out)
    if variant & 2:
        out = _BARE_AGGREGATE.sub(r"\1 AS VAL\2", out)
    if variant & 4:
        out = " ".join(out.split())
    return out


def _mock_summary(prompt: str) -> str:
    match = re.search(r"총\s*(\d+)\s*행", prompt)
    rows = int(match.group(1)) if match else 0
    if rows == 0:
        return "조건에 해당하는 데이터가 없습니다."
    return f"조회 결과 총 {rows}행이 확인되었습니다. (mock 응답)"


def _mock_json(prompt: str) -> str:
    """Answer whichever JSON contract the prompt printed as its output format."""
    if "fixed_sql" in prompt:
        payload: dict[str, object] = {"ok": True, "issue": "", "fixed_sql": ""}
    elif "sub_questions" in prompt:
        payload = {"difficulty": "EASY", "tables": [], "sub_questions": []}
    elif "options" in prompt:
        payload = {"question": "어떤 기준으로 조회할까요?", "options": ["전체", "최근 1년"]}
    else:
        payload = {"ok": True}
    return "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"
