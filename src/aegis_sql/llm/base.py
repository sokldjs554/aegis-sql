"""Provider-agnostic LLM client contract plus cost accounting.

The engine reports a per-query USD cost, and the cascade router uses that
number to decide whether escalating to a frontier model is affordable.  That
only works if every provider funnels through one interface with one price
table, which is what this module is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(slots=True)
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(slots=True)
class LLMResponse:
    text: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    finish_reason: str = "stop"
    raw: dict[str, Any] = field(default_factory=dict)


#: USD per 1M tokens, (input, output).  Update alongside provider pricing.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-fable-5": (1.00, 5.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    # In-house tiers: no marginal API cost, only compute (amortised, ~0).
    "aegis-lm-tiny": (0.0, 0.0),
    "template": (0.0, 0.0),
    "mock": (0.0, 0.0),
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """USD cost for one call; unknown models fall back to a mid-tier estimate."""
    key = next((k for k in PRICING if model.startswith(k)), None)
    inp, out = PRICING.get(key or "", (3.00, 15.00))
    return (prompt_tokens * inp + completion_tokens * out) / 1_000_000.0


@runtime_checkable
class LLMClient(Protocol):
    name: str
    model: str

    def complete(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        """Single completion."""
        ...

    def complete_n(self, messages: list[Message], n: int, **kwargs: Any) -> list[LLMResponse]:
        """``n`` samples — providers that support it batch, others loop."""
        ...

    def available(self) -> bool:
        ...


class LLMUnavailable(RuntimeError):
    """No usable provider (no API key, no local model, network down)."""
