"""Adapter exposing the in-house PyTorch model as a :class:`Generator`.

The model, its tokenizer and its decoding loop live under ``aegis_sql.training``
because that is where they are built; this thin module is what the pipeline
imports, so the generation package presents one uniform surface (template / slm /
llm) and PyTorch is never imported unless the sLLM tier is actually selected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aegis_sql.config import Settings
from aegis_sql.generation.base import GenerationContext
from aegis_sql.observability.logging import get_logger
from aegis_sql.types import GenerationResult, Tier

log = get_logger("generation.slm")


class SLMGenerator:
    """Lazy proxy around ``aegis_sql.training.infer.SLMGenerator``."""

    tier = Tier.SLM
    name = "slm"

    def __init__(self, checkpoint_dir: str | Path, settings: Settings) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.settings = settings
        self._impl: Any | None = None
        self._probed = False

    def _load(self) -> Any | None:
        if self._probed:
            return self._impl
        self._probed = True
        if not self.checkpoint_dir.exists():
            log.debug("sLLM checkpoint absent", path=str(self.checkpoint_dir))
            return None
        try:
            from aegis_sql.training.infer import SLMGenerator as _Impl

            impl = _Impl(self.checkpoint_dir, self.settings)
            self._impl = impl if impl.available() else None
        except Exception as exc:  # torch missing, corrupt checkpoint, ...
            log.info("sLLM tier disabled", reason=str(exc))
            self._impl = None
        return self._impl

    def available(self) -> bool:
        return self._load() is not None

    def generate(self, ctx: GenerationContext) -> GenerationResult:
        impl = self._load()
        if impl is None:
            return GenerationResult(tier=Tier.SLM, model="aegis-lm-tiny (unavailable)")
        return impl.generate(ctx)
