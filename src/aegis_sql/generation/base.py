"""The generator contract.

Three very different things produce SQL in this system — a deterministic
grammar, a 15M-parameter in-house model, and a frontier LLM — and the cascade
router swaps between them per query.  They therefore all implement the same
protocol and return the same :class:`GenerationResult`, so nothing downstream
(verification, repair, voting, tracing, cost accounting) needs to know which
tier answered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from aegis_sql.types import (
    FewShotExample,
    GenerationResult,
    LinkedSchema,
    NormalizedQuestion,
    Tier,
)


@dataclass(slots=True)
class GenerationContext:
    """Everything a generator may look at.  Populated by the pipeline."""

    question: str
    normalized: NormalizedQuestion | None = None
    linked: LinkedSchema | None = None
    #: Pre-rendered schema card (style depends on the tier).
    schema_card: str = ""
    few_shots: list[FewShotExample] = field(default_factory=list)
    #: Reference date for relative time expressions, ``YYYYMMDD``.
    today: str = ""
    dialect: str = "sqlite"
    #: Free-form hints injected by earlier stages (glossary SQL fragments,
    #: decomposition sub-questions, previous failed attempts, ...).
    hints: dict[str, Any] = field(default_factory=dict)
    #: Number of samples to draw (self-consistency).
    n_samples: int = 1
    temperature: float | None = None


@runtime_checkable
class Generator(Protocol):
    """Anything that can turn a question + linked schema into SQL candidates."""

    tier: Tier
    name: str

    def generate(self, ctx: GenerationContext) -> GenerationResult:
        """Produce ``ctx.n_samples`` ordered SQL candidates (best first)."""
        ...

    def available(self) -> bool:
        """Whether this generator can run right now (model present, key set, ...)."""
        ...


class GeneratorError(RuntimeError):
    """Raised when a generator cannot produce any candidate at all."""
