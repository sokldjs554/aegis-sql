"""Hosted frontier providers behind one interface, one retry policy, one price table.

Why this is a module and not ``ChatAnthropic(...)`` at the call site
--------------------------------------------------------------------
* **Billing has to be real.**  The cascade router refuses to escalate when the
  per-query budget is spent, which is only meaningful if the token counts come
  from the provider rather than from an estimator.  Both clients read LangChain's
  normalised ``usage_metadata`` first — for Anthropic that view already folds
  ``cache_read``/``cache_creation`` into ``input_tokens``, which the raw
  ``response_metadata`` does not — then fall back to the raw metadata, and only
  then to a character-based estimate.  Whichever source answered, the number goes
  through the single :func:`~aegis_sql.llm.base.estimate_cost` price table.
* **Retries must be selective.**  Blind retries on a 401 cost three round trips
  and still fail; no retry on a 429 throws away a query that would have
  succeeded.  Errors are classified by HTTP status first and exception name
  second, so the policy works for both SDKs without importing either of them.
  Provider-side retrying is switched off (``max_retries=0``) so that one policy
  owns the latency budget and every attempt is visible in one place.
* **Availability is a two-part predicate.**  A client is usable only if the SDK
  is importable *and* its API key is set.  The import half is answered with
  ``importlib.util.find_spec``, so importing this module never imports LangChain:
  the core engine stays usable — and fast to import — without the ``llm`` extra.

:func:`get_llm_client` is the one entry point and it never raises.  Its last
resort is :class:`~aegis_sql.llm.mock.MockLLM`, which is what keeps CI, the
evaluation harness and a reviewer without an API key running end to end.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from functools import lru_cache
from importlib.util import find_spec
from typing import Any, ClassVar

from tenacity import RetryCallState, Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from aegis_sql.config import Settings, get_settings
from aegis_sql.llm.base import LLMClient, LLMResponse, LLMUnavailable, Message, estimate_cost
from aegis_sql.llm.mock import MockLLM
from aegis_sql.observability.logging import get_logger
from aegis_sql.schema.card import token_estimate
from aegis_sql.types import now_ms

log = get_logger("llm.providers")

#: Statuses that describe "the request was fine, the service was not".
_RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
#: Exception class-name fragments used when no status code is exposed.
_TRANSIENT_MARKERS = (
    "ratelimit", "timeout", "apiconnection", "connection", "internalserver",
    "serviceunavailable", "overloaded", "tryagain",
)
#: Fragments that are never worth a second attempt, whatever the status says.
_FATAL_MARKERS = (
    "authentication", "permissiondenied", "notfound", "badrequest",
    "invalidrequest", "unprocessable", "contentfilter", "apikey",
)


@lru_cache(maxsize=8)
def _module_present(name: str) -> bool:
    """Is the package importable?  Cached — installed packages do not come and go."""
    try:
        return find_spec(name) is not None
    except (ImportError, ValueError):  # namespace weirdness, broken installs
        return False


def _status_of(exc: BaseException) -> int | None:
    for attr in ("status_code", "http_status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _is_transient(exc: BaseException) -> bool:
    """Retry-worthiness of a provider error: rate limits, timeouts and 5xx only."""
    if isinstance(exc, LLMUnavailable):
        return False
    name = type(exc).__name__.lower()
    if any(marker in name for marker in _FATAL_MARKERS):
        return False
    status = _status_of(exc)
    if status is not None:
        return status in _RETRY_STATUS
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    return any(marker in name for marker in _TRANSIENT_MARKERS)


def _log_retry(state: RetryCallState) -> None:  # pragma: no cover - needs a live provider
    exc = state.outcome.exception() if state.outcome else None
    log.warning(
        "transient provider error, retrying",
        attempt=state.attempt_number,
        error=f"{type(exc).__name__}: {exc}" if exc else "?",
    )


def _message_text(message: Any) -> str:
    """Flatten a LangChain ``AIMessage`` into plain text across content-block shapes."""
    text = getattr(message, "text", None)
    if isinstance(text, str):  # langchain-core >= 1.0 exposes a property
        if text:
            return text
    elif callable(text):  # ... and a method before that
        text = text()
        if isinstance(text, str) and text:
            return text
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "".join(parts)


def _to_langchain(messages: Sequence[Message]) -> list[Any]:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    kinds = {"system": SystemMessage, "assistant": AIMessage, "ai": AIMessage}
    return [kinds.get(m.role, HumanMessage)(content=m.content) for m in messages]


_SAMPLING_PARAMS = ("temperature", "top_p", "top_k")

#: sampling 파라미터를 아예 받지 않는 것으로 알려진 모델 (Claude 4.7+/5 계열).
#: 여기에 맞으면 첫 400 왕복조차 없이 처음부터 보내지 않는다.
_NO_SAMPLING_MODEL_PREFIXES = (
    "claude-sonnet-5", "claude-opus-5", "claude-opus-4-7", "claude-opus-4-8",
    "claude-fable", "claude-mythos",
)

#: (provider, model) → 거부된 파라미터 집합. 인스턴스가 아니라 모델 단위로
#: 공유해서, 사전 점검용 클라이언트가 배운 것을 엔진 클라이언트도 물려받는다.
_MODEL_REJECTED_PARAMS: dict[tuple[str, str], set[str]] = {}


def _shared_rejected_params(provider: str, model: str) -> set[str]:
    key = (provider, model)
    if key not in _MODEL_REJECTED_PARAMS:
        seeded = (
            set(_SAMPLING_PARAMS)
            if provider == "anthropic" and model.startswith(_NO_SAMPLING_MODEL_PREFIXES)
            else set()
        )
        _MODEL_REJECTED_PARAMS[key] = seeded
    return _MODEL_REJECTED_PARAMS[key]


def _rejected_sampling_params(exc: Exception) -> set[str]:
    """오류 문구에서 모델이 거부한 sampling 파라미터를 찾아낸다.

    최신 Claude 모델은 ``temperature``/``top_p``/``top_k`` 를 받지 않고 400
    invalid_request_error 로 거절하는데, 문구가 값에 따라 달라진다 —
    "`temperature` is deprecated for this model." 도 있고 value-constraint
    형태도 있다.  특정 문구에만 반응하면 나머지 문구의 400 이 치명 오류로
    분류되어 유효한 키로도 전 문항이 조용히 실패한다 (실측 사고).  그래서
    invalid_request_error 가 sampling 파라미터 이름을 지목하기만 하면 제거
    대상으로 본다 — 제거 후 1회 재시도이므로 오인해도 안전하다.
    """
    text = str(exc)
    if "invalid_request_error" not in text:
        return set()
    return {name for name in _SAMPLING_PARAMS if name in text}


class _LangChainClient:
    """Shared plumbing for the LangChain-backed hosted providers.

    Subclasses declare *what* provider they are (package, env var, model family)
    and how to construct the chat model; everything measurable — retries, usage
    extraction, latency, cost, ``n``-sampling — is implemented once here so the
    two providers cannot drift apart in their accounting.
    """

    provider: ClassVar[str] = ""
    package: ClassVar[str] = ""
    env_var: ClassVar[str] = ""
    default_model: ClassVar[str] = ""
    #: Model-id prefixes that belong to this provider (guards config mismatches).
    model_prefixes: ClassVar[tuple[str, ...]] = ()
    max_attempts: ClassVar[int] = 3

    def __init__(self, settings: Settings | None = None, model: str | None = None) -> None:
        st = settings or get_settings()
        self.settings = st
        self.name = self.provider
        self.model = self._resolve_model(model or st.generation.model)
        self.temperature = float(st.generation.temperature)
        self.max_tokens = int(st.generation.max_tokens)
        self.timeout_s = float(st.generation.request_timeout_s)
        self._chat: Any | None = None
        # 모델이 거부해서 요청에서 빼기로 한 파라미터 — 같은 (provider, model) 의
        # 모든 인스턴스가 한 집합을 공유하고, 알려진 모델은 시드된 상태로 시작한다.
        self._rejected_params: set[str] = _shared_rejected_params(self.provider, self.model)

    # -- capability -------------------------------------------------------- #

    @classmethod
    def usable(cls) -> bool:
        """SDK installed *and* key present — checked without importing the SDK."""
        return _module_present(cls.package) and bool(os.environ.get(cls.env_var, "").strip())

    def available(self) -> bool:
        return type(self).usable()

    @property
    def api_key(self) -> str:
        return os.environ.get(self.env_var, "").strip()

    def _resolve_model(self, model: str) -> str:
        if any(model.lower().startswith(p) for p in self.model_prefixes):
            return model
        log.info(
            "configured model does not belong to this provider, using its default",
            provider=self.provider, configured=model, model=self.default_model,
        )
        return self.default_model

    # -- completion -------------------------------------------------------- #

    def complete(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        chat = self._chat_model()
        payload = _to_langchain(messages)
        overrides: dict[str, Any] = {
            "temperature": self.temperature if temperature is None else float(temperature),
            "max_tokens": int(max_tokens or self.max_tokens),
        }
        if stop:
            overrides["stop"] = list(stop)
        overrides.update(kwargs)
        for name in self._rejected_params:
            overrides.pop(name, None)

        started = now_ms()
        try:
            message = self._retrying()(chat.invoke, payload, **overrides)
        except Exception as exc:
            newly_rejected = _rejected_sampling_params(exc) - self._rejected_params
            if not newly_rejected:
                raise
            # 생성자 kwargs 에도 같은 파라미터가 들어가므로 chat 모델을 다시 만든다.
            self._rejected_params |= newly_rejected
            self._chat = None
            log.warning(
                "model rejects sampling params, retrying without them",
                provider=self.provider, model=self.model,
                params=sorted(self._rejected_params), error=str(exc),
            )
            for name in self._rejected_params:
                overrides.pop(name, None)
            chat = self._chat_model()
            message = self._retrying()(chat.invoke, payload, **overrides)
        latency_ms = now_ms() - started
        response = self._to_response(message, messages, latency_ms)
        log.debug(
            "llm call",
            provider=self.provider, model=response.model,
            prompt_tokens=response.prompt_tokens, completion_tokens=response.completion_tokens,
            latency_ms=round(latency_ms, 1), cost_usd=round(response.cost_usd, 6),
        )
        return response

    def complete_n(
        self,
        messages: list[Message],
        n: int,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> list[LLMResponse]:
        """Draw ``n`` samples by looping.

        Neither provider is batched here: Anthropic has no ``n`` parameter at all,
        and LangChain's ``invoke`` surfaces only the first OpenAI choice, so using
        it would need the lower-level ``generate`` path and would make the two
        providers account for cost differently.  Looping keeps one sample = one
        billed call on both sides.  A failure after the first sample degrades to a
        smaller ensemble instead of losing the whole batch.
        """
        count = max(1, int(n))
        if temperature is None and count > 1:
            # Sampling at temperature 0 returns the same string ``n`` times, which
            # makes self-consistency measure nothing.  (temperature 를 받지 않는
            # 최신 모델에서는 complete() 가 이 값을 빼고 보내고, 모델 기본 온도의
            # 확률적 샘플링이 다양성을 공급한다.)
            temperature = float(self.settings.generation.ensemble_temperature)
        out: list[LLMResponse] = []
        for index in range(count):
            try:
                out.append(self.complete(messages, temperature=temperature, **kwargs))
            except Exception as exc:
                if not out:
                    raise
                log.warning(
                    "sample failed, continuing with a smaller ensemble",
                    provider=self.provider, sample=index, requested=count, error=str(exc),
                )
                break
        return out

    # -- internals --------------------------------------------------------- #

    def _retrying(self) -> Retrying:
        return Retrying(
            retry=retry_if_exception(_is_transient),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8.0),
            stop=stop_after_attempt(self.max_attempts),
            before_sleep=_log_retry,
            reraise=True,
        )

    def _chat_model(self) -> Any:
        if self._chat is None:
            if not self.available():
                raise LLMUnavailable(
                    f"{self.provider} 사용 불가: {self.env_var} 미설정 또는 {self.package} 미설치"
                )
            self._chat = self._build()
            log.debug("chat model constructed", provider=self.provider, model=self.model)
        return self._chat

    def _build(self) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError

    def _chat_kwargs(self) -> dict[str, Any]:
        """Constructor arguments both chat models accept.

        These are LangChain field *aliases* rather than the SDKs' own parameter
        names, which is exactly why they are written once: the two adapters spell
        the underlying options differently (``max_tokens_to_sample`` versus
        ``max_completion_tokens``, ``default_request_timeout`` versus
        ``request_timeout``) and LangChain normalises both.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout_s,
            "max_retries": 0,  # tenacity owns the retry policy
        }
        for name in self._rejected_params:
            kwargs.pop(name, None)
        return kwargs

    def _to_response(
        self, message: Any, request: Sequence[Message], latency_ms: float
    ) -> LLMResponse:
        text = _message_text(message)
        meta = dict(getattr(message, "response_metadata", None) or {})
        usage = dict(getattr(message, "usage_metadata", None) or {})
        prompt_tokens, completion_tokens = _usage(usage, meta, request, text)
        model = str(meta.get("model_name") or meta.get("model") or self.model)
        return LLMResponse(
            text=text,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            cost_usd=estimate_cost(model, prompt_tokens, completion_tokens),
            finish_reason=str(meta.get("stop_reason") or meta.get("finish_reason") or "stop"),
            raw={"provider": self.provider, "usage": usage, "metadata": meta},
        )


def _usage(
    usage: dict[str, Any], meta: dict[str, Any], request: Sequence[Message], completion: str
) -> tuple[int, int]:
    """Token counts, preferring the provider's own numbers over any estimate."""
    prompt_tokens = usage.get("input_tokens")
    completion_tokens = usage.get("output_tokens")
    raw = meta.get("usage") or meta.get("token_usage") or {}
    if not isinstance(raw, dict):
        raw = {}
    if prompt_tokens is None:
        prompt_tokens = raw.get("input_tokens") or raw.get("prompt_tokens")
    if completion_tokens is None:
        completion_tokens = raw.get("output_tokens") or raw.get("completion_tokens")
    if prompt_tokens is None:
        prompt_tokens = token_estimate("\n".join(m.content for m in request))
    if completion_tokens is None:
        completion_tokens = token_estimate(completion)
    return int(prompt_tokens), int(completion_tokens)


class AnthropicClient(_LangChainClient):
    """Claude through ``langchain_anthropic.ChatAnthropic``."""

    provider = "anthropic"
    package = "langchain_anthropic"
    env_var = "ANTHROPIC_API_KEY"
    default_model = "claude-sonnet-5"
    model_prefixes = ("claude",)

    def _build(self) -> Any:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(**self._chat_kwargs())


class OpenAIClient(_LangChainClient):
    """GPT through ``langchain_openai.ChatOpenAI``."""

    provider = "openai"
    package = "langchain_openai"
    env_var = "OPENAI_API_KEY"
    default_model = "gpt-4o"
    model_prefixes = ("gpt", "o1", "o3", "o4", "chatgpt")

    def _build(self) -> Any:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(**self._chat_kwargs())


#: Ordered by preference for ``provider: auto``.
_PROVIDERS: tuple[type[_LangChainClient], ...] = (AnthropicClient, OpenAIClient)
_ALIASES = {
    "anthropic": "anthropic", "claude": "anthropic",
    "openai": "openai", "gpt": "openai",
    "mock": "mock",
}


def available_providers() -> dict[str, bool]:
    """``{provider: usable right now}`` — served by ``/v1/health`` and ``aegis version``."""
    providers = {cls.provider: cls.usable() for cls in _PROVIDERS}
    providers["mock"] = True
    return providers


def get_llm_client(settings: Settings, prefer: str | None = None) -> LLMClient:
    """Pick a client.  Never raises; falls back to the offline mock.

    ``prefer`` accepts ``auto`` (or ``None``), ``anthropic``/``claude``,
    ``openai``/``gpt`` and ``mock``.  Anything else — ``template``, ``slm``, a
    typo — is treated as ``auto``, because an unrecognised preference should not
    silently disable the tier that answers repair and answer-synthesis calls.
    """
    requested = (prefer or "auto").strip().lower()
    choice = _ALIASES.get(requested)
    if choice is None and requested not in {"auto", ""}:
        log.debug("unrecognised provider preference, treating as auto", prefer=prefer)

    if choice == "mock":
        return _mock_client("provider=mock 로 명시 요청됨")

    if choice is not None:
        cls = next(c for c in _PROVIDERS if c.provider == choice)
        if cls.usable():
            return _selected(cls(settings), "explicit")
        log.warning(
            "requested provider is unavailable, falling back",
            provider=choice, env_var=cls.env_var, installed=_module_present(cls.package),
        )

    for cls in _PROVIDERS:
        if cls.usable():
            return _selected(cls(settings), "auto")
    return _mock_client("API 키가 없어 오프라인 mock 으로 동작합니다")


def _selected(client: _LangChainClient, reason: str) -> LLMClient:
    log.info("llm provider selected", provider=client.name, model=client.model, reason=reason)
    return client


def _mock_client(reason: str) -> LLMClient:
    log.info("llm provider selected", provider="mock", model="mock", reason=reason)
    return MockLLM()
