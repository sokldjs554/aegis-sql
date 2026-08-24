"""Serving adapter for the in-house model — the sLLM tier of the cascade.

This is the only module in :mod:`aegis_sql.training` the request path touches,
so it is written defensively around one rule: **the sLLM tier must never be the
reason a query fails.**  A missing checkpoint, a torch-less install or a corrupt
weights file makes :meth:`SLMGenerator.available` return ``False`` and
:meth:`SLMGenerator.generate` return an empty :class:`GenerationResult`; the
router then falls through to the LLM tier and the user sees an answer instead of
a stack trace.  Nothing here raises.

Two things it does that a naive decode loop does not:

* **The prompt is built by** :func:`aegis_sql.training.sft.build_prompt`, the
  exact function the training data was rendered with, and the ``<|sql|>``
  boundary is appended as a *token id* rather than as text.  Train/serve prompt
  drift is the single most common way a small model quietly loses accuracy after
  a refactor, and importing the one definition makes it impossible.
* **Candidates are scored, not just produced.**  After decoding, one batched
  forward pass computes the mean token log-probability of each candidate over
  the generated span only.  That number orders the candidates for
  self-consistency voting downstream and gives the router a real confidence
  signal instead of "the model said something".

Everything torch-related is imported inside :meth:`_load`, so importing this
module costs nothing and the core engine stays installable without the training
extra.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from aegis_sql.generation.base import GenerationContext
from aegis_sql.observability.logging import get_logger
from aegis_sql.types import GenerationResult, SQLCandidate, Tier

log = get_logger("training.infer")

CONFIG_FILE = "config.json"
WEIGHTS_FILE = "model.pt"
TOKENIZER_FILE = "tokenizer.json"

#: Hard ceiling on decoded SQL length; the training corpus p99 is well under this.
DEFAULT_MAX_NEW_TOKENS = 256


class SLMGenerator:
    """Implements the :class:`~aegis_sql.generation.base.Generator` protocol."""

    tier = Tier.SLM
    name = "slm"

    def __init__(self, checkpoint_dir: str | Path, settings: Any) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.settings = settings
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._model_name = "aegis-lm"
        self._load_failed = False

    # -- availability ------------------------------------------------------- #

    def _files_present(self) -> bool:
        d = self.checkpoint_dir
        return all((d / f).exists() for f in (CONFIG_FILE, WEIGHTS_FILE, TOKENIZER_FILE))

    def available(self) -> bool:
        """True when the checkpoint is complete and torch can actually load it."""
        if self._load_failed or not self._files_present():
            return False
        if self._model is not None:
            return True
        try:
            from aegis_sql.training.model import TORCH_AVAILABLE
        except Exception:  # pragma: no cover - broken install
            return False
        return bool(TORCH_AVAILABLE)

    def _load(self) -> bool:
        """Lazily materialise the model.  Returns False instead of raising."""
        if self._model is not None and self._tokenizer is not None:
            return True
        if self._load_failed or not self._files_present():
            return False
        try:
            from aegis_sql.training.model import AegisLM
            from aegis_sql.training.tokenizer import ByteBPETokenizer

            model = AegisLM.from_pretrained(self.checkpoint_dir)
            model.eval()
            self._model = model
            self._tokenizer = ByteBPETokenizer.load(self.checkpoint_dir)
            params = model.num_parameters()
            self._model_name = f"aegis-lm-{params / 1e6:.1f}M"
            log.info(
                "sLLM loaded",
                path=str(self.checkpoint_dir),
                params=params,
                vocab=self._tokenizer.vocab_size,
                max_seq_len=model.cfg.max_seq_len,
            )
            return True
        except Exception as exc:
            self._load_failed = True
            log.warning(
                "sLLM load failed", path=str(self.checkpoint_dir), error=f"{type(exc).__name__}: {exc}"
            )
            return False

    # -- generation --------------------------------------------------------- #

    def generate(self, ctx: GenerationContext) -> GenerationResult:
        started = time.perf_counter()
        if not self._load():
            return GenerationResult(tier=Tier.SLM, model=f"{self._model_name} (unavailable)")
        try:
            return self._generate(ctx, started)
        except Exception as exc:  # decoding must never break the request path
            log.warning("sLLM generation failed", error=f"{type(exc).__name__}: {exc}")
            return GenerationResult(
                tier=Tier.SLM,
                model=self._model_name,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

    def _generate(self, ctx: GenerationContext, started: float) -> GenerationResult:
        import torch

        from aegis_sql.training.sft import build_prompt

        model: Any = self._model
        tok: Any = self._tokenizer
        gen_cfg = getattr(self.settings, "generation", None)

        n_samples = max(1, ctx.n_samples)
        temperature = ctx.temperature
        if temperature is None:
            temperature = (
                getattr(gen_cfg, "ensemble_temperature", 0.7)
                if n_samples > 1
                else getattr(gen_cfg, "temperature", 0.0)
            )
        cap = int(getattr(gen_cfg, "slm_max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
        max_new = min(int(getattr(gen_cfg, "max_tokens", DEFAULT_MAX_NEW_TOKENS)), cap)

        prompt = build_prompt(ctx.question, ctx.schema_card)
        prompt_ids = tok.encode(prompt, add_bos=True)
        budget = model.cfg.max_seq_len - max_new - 1
        if len(prompt_ids) > budget:  # left-truncate exactly as SFT did
            prompt_ids = [tok.bos_id] + prompt_ids[1:][-(budget - 1) :]
        prompt_ids = prompt_ids + [tok.sep_id]

        seed = getattr(getattr(self.settings, "training", None), "seed", 20260824)
        torch.manual_seed(seed)
        inputs = torch.tensor([prompt_ids], dtype=torch.long).repeat(n_samples, 1)
        output = model.generate(
            inputs,
            max_new_tokens=max_new,
            temperature=float(temperature),
            top_p=0.95 if temperature and temperature > 0 else 1.0,
            eos_id=tok.eos_id,
            use_cache=True,
        )

        completions = [_cut_at_eos(row.tolist()[len(prompt_ids) :], tok.eos_id) for row in output]
        scores = _score_completions(model, prompt_ids, completions, tok.pad_id)

        seen: dict[str, SQLCandidate] = {}
        for ids, logprob in zip(completions, scores, strict=True):
            sql = _clean_sql(tok.decode(ids))
            if not sql:
                continue
            candidate = SQLCandidate(
                sql=sql,
                tier=Tier.SLM,
                logprob=logprob,
                raw_output=tok.decode(ids),
                prompt_version=f"slm/{getattr(gen_cfg, 'prompt_set', 'default')}",
            )
            key = candidate.normalized_key()
            existing = seen.get(key)
            if existing is None:
                seen[key] = candidate
                candidate.votes = 1
            else:
                existing.votes += 1
                if logprob is not None and (existing.logprob is None or logprob > existing.logprob):
                    existing.logprob = logprob

        candidates = sorted(seen.values(), key=lambda c: (-c.votes, -(c.logprob or -1e9)))
        return GenerationResult(
            candidates=candidates,
            tier=Tier.SLM,
            model=self._model_name,
            prompt_tokens=len(prompt_ids),
            completion_tokens=sum(len(c) for c in completions),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            cost_usd=0.0,
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _cut_at_eos(ids: list[int], eos_id: int) -> list[int]:
    return ids[: ids.index(eos_id)] if eos_id in ids else ids


def _clean_sql(text: str) -> str:
    """Normalise whitespace and drop a trailing semicolon so sqlglot sees one statement."""
    sql = text.strip()
    if sql.endswith(";"):
        sql = sql[:-1].rstrip()
    return " ".join(sql.split()) if "\n" not in sql else sql


def _score_completions(
    model: Any, prompt_ids: list[int], completions: list[list[int]], pad_id: int
) -> list[float | None]:
    """Mean token log-probability of each completion under the model."""
    import torch

    from aegis_sql.training.sft import IGNORE_INDEX

    usable = [c for c in completions if c]
    if not usable:
        return [None] * len(completions)

    width = len(prompt_ids) + max(len(c) for c in usable)
    input_rows, label_rows, mask_rows = [], [], []
    for ids in completions:
        body = ids or [pad_id]
        row = prompt_ids + body
        pad = width - len(row)
        input_rows.append(row + [pad_id] * pad)
        label_rows.append([IGNORE_INDEX] * len(prompt_ids) + body + [IGNORE_INDEX] * pad)
        mask_rows.append([1] * len(row) + [0] * pad)

    with torch.no_grad():
        out = model(
            input_ids=torch.tensor(input_rows, dtype=torch.long),
            attention_mask=torch.tensor(mask_rows, dtype=torch.long),
        )
        labels = torch.tensor(label_rows, dtype=torch.long)[:, 1:]
        logits = out["logits"][:, :-1, :]
        keep = labels != IGNORE_INDEX
        token_logp = logits.log_softmax(-1).gather(-1, labels.masked_fill(~keep, 0).unsqueeze(-1))
        totals = (token_logp.squeeze(-1) * keep).sum(-1)
        counts = keep.sum(-1).clamp(min=1)
        means = (totals / counts).tolist()

    return [float(m) if ids else None for m, ids in zip(means, completions, strict=True)]
