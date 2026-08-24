"""The frontier tier: registry prompts in, one statement out, billed and traced.

What this module actually has to get right
------------------------------------------
Calling a hosted model is the easy part.  The three things that decide whether
this tier is usable in production are:

1. **Getting exactly one statement back out of prose.**  The system prompt asks
   for a single fenced block, and models still return a diagnosis paragraph, two
   candidate blocks, a block truncated by ``max_tokens``, or bare SQL with a
   trailing Korean sentence.  :func:`extract_sql` is therefore a module-level
   pure function with its own unit tests, not a lambda inside the call path: it
   prefers the *last* ```` ```sql ```` block (models revise, and the revision is
   last), accepts an unterminated fence, and falls back to a scan from the first
   ``SELECT``/``WITH`` to the statement-ending semicolon — with Korean prose
   lines dropped only when they contain no quote and no SQL token, so that
   ``WHERE PROD_NM = '종신보험'`` survives.
2. **Never taking the pipeline down.**  A 429, an expired key or a garbled
   completion must not raise: :meth:`LLMGenerator.generate` returns an empty
   :class:`~aegis_sql.types.GenerationResult`, which the cascade reads as "this
   tier produced nothing" and escalates or falls back to the template tier.  The
   auxiliary calls (:meth:`repair`, :meth:`synthesize_answer`,
   :meth:`self_check`) degrade to ``None`` / ``""`` / ``{}`` for the same reason.
3. **Billing every token.**  Prompt/completion tokens and USD from every sample
   are accumulated onto the result, and each candidate is stamped with
   ``prompt_version = registry.get('nl2sql.user').ref`` so an evaluation number
   can always be traced back to the exact prompt text that produced it.

Where LangChain earns its place
-------------------------------
LangChain is used for two things and deliberately not for a third.  It provides
the *provider adapters* (:mod:`aegis_sql.llm.providers`) and the *composition
surface*: :meth:`LLMGenerator.as_runnable` exposes this tier as an LCEL chain,

    prompt values → ``RunnableLambda(_to_messages)`` → ``RunnableLambda(_invoke)``
    → ``RunnableLambda(parse_completion)`` → SQL

so the generator drops into a larger graph (``linker | generator | guard``) and
inherits callbacks/LangSmith tracing, ``batch``/``stream``, ``with_retry`` and
``with_fallbacks`` for free.  The chain carries an :class:`LLMCompletion` — SQL
*plus* the response — right up to the last step, because a parser that returns
only a string throws away the token counts the budget depends on.

What LangChain is not used for: prompt management (the versioned
:class:`~aegis_sql.prompts.registry.PromptRegistry` is the source of truth, and
hashed prompt refs are worth more than ``ChatPromptTemplate``) and output
parsing (see point 1).  And when ``langchain_core`` is not installed at all the
same three functions are called directly, so the tier behaves identically minus
the callbacks.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from aegis_sql.config import Settings
from aegis_sql.generation.base import GenerationContext
from aegis_sql.llm.base import LLMClient, LLMResponse, Message
from aegis_sql.observability.logging import get_logger
from aegis_sql.prompts.registry import PromptRegistry
from aegis_sql.types import GenerationResult, SQLCandidate, Tier, now_ms

log = get_logger("generation.llm")

#: A fenced block that declares a SQL dialect.  ``(?:```|$)`` also accepts a block
#: truncated by ``max_tokens`` — a long statement is still worth recovering.
_SQL_FENCE = re.compile(
    r"```[ \t]*(?:sql|sqlite|postgres(?:ql)?|mysql|tsql|bigquery)\b[ \t]*\r?\n?(.*?)(?:```|$)",
    re.DOTALL | re.IGNORECASE,
)
_ANY_FENCE = re.compile(r"```[a-zA-Z]*[ \t]*\r?\n?(.*?)(?:```|$)", re.DOTALL)
_SQL_START = re.compile(r"\b(?:WITH|SELECT)\b", re.IGNORECASE)
_HANGUL = re.compile(r"[가-힣]")
_SQL_TOKEN = re.compile(
    r"(?i)\b(?:select|from|where|group|order|having|join|on|and|or|union|with|limit|offset|as|"
    r"case|when|then|else|end|by|desc|asc|left|right|inner|outer|not|in|between|like|is|null|"
    r"distinct|count|sum|avg|min|max|cast|substr|coalesce|nullif|over|partition)\b|[(),*=<>]"
)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


# --------------------------------------------------------------------------- #
# SQL extraction (pure, unit-tested)
# --------------------------------------------------------------------------- #


def extract_sql(text: str) -> str | None:
    """Recover one SQL statement from a model completion, or ``None``.

    Resolution order: last dialect-tagged fence → last untagged fence → scan from
    the first ``SELECT``/``WITH``.  Trailing semicolons and trailing prose are
    stripped; a block that does not begin with ``SELECT``/``WITH`` is skipped
    rather than returned, so a ```` ```json ```` verdict block is never mistaken
    for a query.
    """
    if not text or not text.strip():
        return None

    for pattern in (_SQL_FENCE, _ANY_FENCE):
        for block in reversed([m.group(1) for m in pattern.finditer(text)]):
            sql = _clean_block(block)
            if sql:
                return sql

    match = _SQL_START.search(text)
    if match is None:
        return None
    return _clean_block(text[match.start() :])


def _clean_block(block: str) -> str | None:
    match = _SQL_START.search(block)
    if match is None:
        return None
    body = _cut_at_statement_end(block[match.start() :])
    body = _drop_trailing_prose(body)
    sql = body.strip().rstrip(";").strip()
    return sql or None


def _cut_at_statement_end(text: str) -> str:
    """Truncate at the first ``;`` that is not inside a string literal."""
    in_string = False
    for index, char in enumerate(text):
        if char == "'":
            in_string = not in_string
        elif char == ";" and not in_string:
            return text[:index]
    return text


def _drop_trailing_prose(text: str) -> str:
    kept: list[str] = []
    for line in text.splitlines():
        if kept and _is_prose(line):
            break
        kept.append(line)
    return "\n".join(kept)


def _is_prose(line: str) -> bool:
    """Korean commentary, as opposed to SQL that happens to contain Korean literals."""
    stripped = line.strip()
    if not stripped or stripped.startswith(("--", "/*", "*")):
        return False
    if not _HANGUL.search(stripped):
        return False
    if "'" in stripped or '"' in stripped:  # a Korean value literal
        return False
    return not _SQL_TOKEN.search(stripped)


# --------------------------------------------------------------------------- #
# What the LCEL chain carries
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class LLMCompletion:
    """One completion and the SQL parsed out of it.

    The chain passes this rather than a bare string so that token counts and cost
    survive the parsing step — the router's budget is computed from them.
    """

    sql: str | None
    response: LLMResponse


def parse_completion(response: LLMResponse) -> LLMCompletion:
    """Chain step: :class:`LLMResponse` → :class:`LLMCompletion`."""
    return LLMCompletion(sql=extract_sql(response.text), response=response)


def parse_sql(payload: LLMResponse | LLMCompletion | str) -> str | None:
    """Chain step: anything the invoke step can return → SQL text."""
    if isinstance(payload, LLMCompletion):
        return payload.sql
    if isinstance(payload, LLMResponse):
        return extract_sql(payload.text)
    return extract_sql(payload)


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #


class LLMGenerator:
    """Frontier-model tier implementing :class:`~aegis_sql.generation.base.Generator`."""

    tier = Tier.LLM
    name = "llm"

    def __init__(self, client: LLMClient, registry: PromptRegistry, settings: Settings) -> None:
        self.client = client
        self.registry = registry
        self.settings = settings
        #: Reference date of the query being served, mirrored from the last
        #: ``GenerationContext`` so that repair and self-check resolve relative
        #: dates exactly as generation did.
        self.today = ""
        #: USD spent on repair / answer / self-check calls, which the pipeline
        #: bills to the ``generate`` span; exposed so an eval run can report the
        #: true per-query cost.
        self.aux_cost_usd = 0.0
        self._chain: Any | None = None
        self._chain_probed = False

    # -- capability -------------------------------------------------------- #

    def available(self) -> bool:
        """Usable *and* worth using.

        A :class:`~aegis_sql.llm.mock.MockLLM` is always usable, but advertising
        the LLM tier while a fixture answers would make the cascade prefer canned
        SQL over the template tier's semantic parse.  The mock therefore counts
        only when it was asked for by name (``generation.provider: mock``), which
        is what makes an offline end-to-end demo possible without making it the
        accidental default.
        """
        try:
            if not self.client.available():
                return False
        except Exception as exc:  # a provider probe must never break engine startup
            log.debug("provider probe failed", error=str(exc))
            return False
        if getattr(self.client, "name", "") == "mock":
            return self.settings.generation.provider == "mock"
        return True

    # -- generation -------------------------------------------------------- #

    def generate(self, ctx: GenerationContext) -> GenerationResult:
        n = max(1, int(ctx.n_samples))
        tier = Tier.ENSEMBLE if n > 1 else Tier.LLM
        model = str(getattr(self.client, "model", "") or getattr(self.client, "name", "llm"))
        result = GenerationResult(tier=tier, model=model)
        if not self.available():
            log.debug("llm tier unavailable, returning no candidates", model=model)
            return result

        self.today = ctx.today or self.today
        started = now_ms()
        try:
            completions = self._complete(ctx, n)
        except Exception as exc:
            # Escalation, not exceptions: an empty result lets the cascade fall
            # back to a tier that can answer.
            log.error("llm generation failed", model=model, samples=n, error=str(exc))
            result.latency_ms = now_ms() - started
            return result

        version = self.registry.get("nl2sql.user").ref
        for completion in completions:
            response = completion.response
            result.prompt_tokens += response.prompt_tokens
            result.completion_tokens += response.completion_tokens
            result.cost_usd += response.cost_usd
            if response.model:
                result.model = response.model
            if not completion.sql:
                log.warning("completion contained no SQL", finish=response.finish_reason)
                continue
            result.candidates.append(
                SQLCandidate(
                    sql=completion.sql,
                    tier=tier,
                    raw_output=response.text,
                    prompt_version=version,
                )
            )
        result.latency_ms = now_ms() - started
        log.info(
            "llm generation",
            model=result.model, samples=len(completions), candidates=len(result.candidates),
            prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens,
            cost_usd=round(result.cost_usd, 6), latency_ms=round(result.latency_ms, 1),
        )
        return result

    def _complete(self, ctx: GenerationContext, n: int) -> list[LLMCompletion]:
        if n == 1:
            return [self._run(ctx)]
        # ``complete_n`` is the provider's sampling contract (see providers.py);
        # the chain covers the single-sample path.
        payload = self._to_messages(ctx)
        responses = self.client.complete_n(
            payload["messages"], n, temperature=payload["temperature"]
        )
        return [parse_completion(r) for r in responses]

    def _run(self, source: GenerationContext | Mapping[str, Any]) -> LLMCompletion:
        """One completion through the LCEL chain, or through the same steps directly."""
        chain = self._core_chain()
        if chain is not None:
            return chain.invoke(source)
        return parse_completion(self._invoke(self._to_messages(source)))

    # -- LCEL -------------------------------------------------------------- #

    def _core_chain(self) -> Any | None:
        """``prompt values | invoke | parse`` — ``None`` when langchain is absent."""
        if not self._chain_probed:
            self._chain_probed = True
            try:
                from langchain_core.runnables import RunnableLambda

                self._chain = (
                    RunnableLambda(self._to_messages).with_config(run_name="aegis.prompt")
                    | RunnableLambda(self._invoke).with_config(run_name="aegis.llm")
                    | RunnableLambda(parse_completion).with_config(run_name="aegis.parse")
                )
            except Exception as exc:  # langchain_core not installed
                log.info("LCEL unavailable, using the direct call path", reason=str(exc))
                self._chain = None
        return self._chain

    def as_runnable(self) -> Any:
        """This tier as an LCEL ``Runnable``: prompt values in, SQL string out.

        Accepts a :class:`~aegis_sql.generation.base.GenerationContext` or a plain
        mapping of prompt values, so it composes into a larger graph.  Raises
        ``ImportError`` when ``langchain_core`` is missing — callers that must work
        without it should use :meth:`generate`.
        """
        from langchain_core.runnables import RunnableLambda

        chain = self._core_chain()
        if chain is None:  # pragma: no cover - only when the import above succeeded
            raise ImportError("langchain_core is required for as_runnable()")
        return chain | RunnableLambda(parse_sql).with_config(run_name="aegis.sql")

    # -- prompt construction ----------------------------------------------- #

    def prompt_values(self, ctx: GenerationContext) -> dict[str, Any]:
        """The variables ``nl2sql.system`` / ``nl2sql.user`` are rendered with."""
        linked = ctx.linked
        hints = ctx.hints or {}
        return {
            "dialect": ctx.dialect or self.settings.database.dialect,
            "today": ctx.today or self._today(),
            "schema_card": ctx.schema_card,
            "question": ctx.question,
            "glossary": list(linked.glossary) if linked else [],
            "few_shots": list(ctx.few_shots),
            "hints": _hint_lines(hints),
            "failed_attempts": hints.get("failed_attempts") or [],
            "temperature": ctx.temperature,
            "max_tokens": self.settings.generation.max_tokens,
        }

    def build_messages(self, values: Mapping[str, Any]) -> list[Message]:
        """Render the registry prompts into the provider-agnostic message list."""
        system = self.registry.render(
            "nl2sql.system",
            dialect=values.get("dialect") or self.settings.database.dialect,
            today=values.get("today") or self._today(),
        )
        user = self.registry.render(
            "nl2sql.user",
            schema_card=values.get("schema_card", ""),
            question=values.get("question", ""),
            glossary=values.get("glossary") or [],
            few_shots=values.get("few_shots") or [],
            hints=values.get("hints") or [],
            failed_attempts=values.get("failed_attempts") or [],
        )
        return [Message("system", system), Message("user", user)]

    def _to_messages(self, payload: GenerationContext | Mapping[str, Any]) -> dict[str, Any]:
        """Chain step: a context, prompt values, or a ready payload → the invoke payload.

        Accepting an already-rendered payload keeps the step idempotent, so a
        caller that composes :meth:`as_runnable` after its own rendering step does
        not silently re-render an empty prompt.
        """
        if isinstance(payload, Mapping) and "messages" in payload:
            values: dict[str, Any] = dict(payload)
            return {
                "messages": list(values["messages"]),
                "temperature": values.get("temperature", self.settings.generation.temperature),
                "max_tokens": int(values.get("max_tokens") or self.settings.generation.max_tokens),
            }
        values = (
            self.prompt_values(payload)
            if isinstance(payload, GenerationContext)
            else dict(payload)
        )
        temperature = values.get("temperature")
        return {
            "messages": self.build_messages(values),
            "temperature": (
                float(temperature) if temperature is not None else self.settings.generation.temperature
            ),
            "max_tokens": int(values.get("max_tokens") or self.settings.generation.max_tokens),
        }

    def _invoke(self, payload: Mapping[str, Any]) -> LLMResponse:
        """Chain step: the payload → one completion."""
        return self.client.complete(
            list(payload["messages"]),
            temperature=payload.get("temperature"),
            max_tokens=payload.get("max_tokens"),
        )

    # -- auxiliary model calls --------------------------------------------- #

    def repair(self, rctx: Any) -> str | None:
        """Model-backed last-resort repair (``repair.user``).

        Duck-typed on :class:`~aegis_sql.verify.repair.RepairContext` so the
        generation package does not import the verification package.  Returns
        ``None`` — never raises — when the tier is off or the answer is unusable,
        which the rule-based repairer treats as "this strategy did not fire".
        """
        if not self.available():
            return None
        try:
            user = self.registry.render(
                "repair.user",
                schema_card=getattr(rctx, "schema_card", "") or "",
                question=getattr(rctx, "question", "") or "",
                sql=getattr(rctx, "sql", "") or "",
                error=getattr(rctx, "error", "") or "",
                attempted_fixes=getattr(rctx, "attempted_fixes", None) or [],
            )
            response = self._ask(
                [Message("system", self._system_prompt()), Message("user", user)]
            )
            sql = extract_sql(response.text)
            log.info(
                "llm repair",
                attempt=getattr(rctx, "attempt", 0), fixed=sql is not None,
                cost_usd=round(response.cost_usd, 6),
            )
            return sql
        except Exception as exc:
            log.warning("llm repair failed", error=str(exc))
            return None

    def synthesize_answer(
        self,
        question: str,
        sql: str,
        rows: Sequence[Sequence[Any]],
        columns: Sequence[str],
        row_count: int,
    ) -> str:
        """Korean natural-language summary of a result set (``answer.user``).

        No system prompt: ``nl2sql.system`` orders the model to emit nothing but
        SQL, which is precisely wrong here.  Returns ``""`` on any failure, and
        the pipeline falls back to its deterministic Korean formatter.
        """
        if not self.available():
            return ""
        try:
            user = self.registry.render(
                "answer.user",
                question=question,
                sql=sql,
                result_table=render_result_table(columns, rows),
                row_count=row_count,
            )
            response = self._ask([Message("user", user)])
            return _strip_fences(response.text).strip()
        except Exception as exc:
            log.warning("answer synthesis failed", error=str(exc))
            return ""

    def self_check(self, question: str, sql: str, schema_card: str) -> dict[str, Any]:
        """Pre-execution critic (``selfcheck.user``).

        Returns ``{"ok": bool, "issue": str, "fixed_sql": str}``, or ``{}`` when
        the tier is off or the model did not answer with parseable JSON — an
        empty dict means "no opinion", never "looks fine".
        """
        if not self.available():
            return {}
        try:
            user = self.registry.render(
                "selfcheck.user", question=question, sql=sql, schema_card=schema_card
            )
            response = self._ask([Message("user", user)])
            data = _parse_json_object(response.text)
            if not isinstance(data, dict):
                log.warning("self-check returned no JSON object")
                return {}
            verdict = {
                "ok": bool(data.get("ok", True)),
                "issue": str(data.get("issue") or ""),
                "fixed_sql": str(data.get("fixed_sql") or ""),
            }
            log.info("llm self-check", ok=verdict["ok"], has_fix=bool(verdict["fixed_sql"]))
            return verdict
        except Exception as exc:
            log.warning("self-check failed", error=str(exc))
            return {}

    # -- helpers ----------------------------------------------------------- #

    def _ask(self, messages: list[Message]) -> LLMResponse:
        """One auxiliary call at temperature 0, billed to :attr:`aux_cost_usd`."""
        response = self.client.complete(messages, temperature=0.0)
        self.aux_cost_usd += response.cost_usd
        return response

    def _system_prompt(self) -> str:
        return self.registry.render(
            "nl2sql.system", dialect=self.settings.database.dialect, today=self._today()
        )

    def _today(self) -> str:
        # Mirrors the reference date the pipeline passed in; only a standalone
        # caller that never ran generate() reaches the wall clock.
        return self.today or date.today().strftime("%Y%m%d")


# --------------------------------------------------------------------------- #
# Module-level helpers
# --------------------------------------------------------------------------- #


def render_result_table(
    columns: Sequence[str], rows: Sequence[Sequence[Any]], max_rows: int = 20, max_cell: int = 60
) -> str:
    """Pipe-delimited rendering of a result set for the answer prompt."""
    if not columns:
        return "(결과 없음)"
    lines = [" | ".join(str(c) for c in columns)]
    lines.append("-|-".join("-" * len(str(c)) for c in columns))
    for row in list(rows)[:max_rows]:
        cells = []
        for value in row:
            cell = "NULL" if value is None else str(value)
            cells.append(cell if len(cell) <= max_cell else cell[: max_cell - 1] + "…")
        lines.append(" | ".join(cells))
    if len(rows) > max_rows:
        lines.append(f"... ({len(rows) - max_rows}행 생략)")
    return "\n".join(lines)


def _hint_lines(hints: Mapping[str, Any]) -> list[str]:
    """Flatten the pipeline's free-form hint bag into prompt lines."""
    lines: list[str] = []
    value = hints.get("hints")
    if isinstance(value, str):
        lines.append(value)
    elif isinstance(value, (list, tuple)):
        lines.extend(str(item) for item in value if str(item).strip())
    sub_questions = hints.get("sub_questions")
    if isinstance(sub_questions, (list, tuple)):
        lines.extend(f"서브질문 {i + 1}: {q}" for i, q in enumerate(sub_questions))
    return lines


def _strip_fences(text: str) -> str:
    match = _ANY_FENCE.search(text)
    return match.group(1) if match and match.group(1).strip() else text


def _parse_json_object(text: str) -> Any:
    match = _JSON_OBJECT.search(_strip_fences(text))
    if match is None:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
