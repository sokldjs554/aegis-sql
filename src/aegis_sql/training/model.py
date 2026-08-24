"""The in-house decoder-only Transformer (``aegis-lm``), written out in full.

The cheap tier of the cascade cannot be a downloaded checkpoint.  A Korean
insurance schema is proprietary, the deployment target is an on-premise CPU box
behind a bank firewall with no model hub reachable, and the tokenizer has to be
fitted on the domain corpus (see :mod:`aegis_sql.training.tokenizer`) — which
already rules out reusing someone else's embedding matrix.  So the model is
built here, from ``nn.Linear`` up, in roughly 250 lines.

The architecture is the 2024-era decoder consensus, chosen because each piece
earns its place at this scale:

* **Pre-norm + RMSNorm.**  No mean subtraction and no bias, so it is cheaper
  than LayerNorm and, more importantly, pre-norm keeps the residual stream a
  clean identity path — which is what makes a 6-layer model trainable at
  ``lr=3e-4`` without a warmup babysitter.
* **RoPE** instead of learned positional embeddings.  Relative by construction,
  it extrapolates gracefully when a schema card pushes a sequence past the
  lengths seen in training, and it costs no parameters — meaningful when the
  whole model is 15M.
* **SwiGLU** feed-forward: ~2 points of loss over ReLU MLP at equal parameter
  count, which is the cheapest quality anyone is handing out.
* **Tied embeddings.**  With an 8k vocabulary and ``d_model=256``, the embedding
  matrix is a *sixth* of the model.  Tying it to the output head removes that
  duplicate and measurably reduces overfitting on a synthetic corpus.

Projections are deliberately named ``q_proj``/``k_proj``/``v_proj``/``o_proj``
and ``gate_proj``/``up_proj``/``down_proj`` so that
:func:`aegis_sql.training.lora.apply_lora` can target them by name, exactly the
way adapters are targeted on a real Llama-family checkpoint.

torch is an optional extra (``pip install aegis-sql[train]``).  This module is
therefore importable without it — the classes are still defined, and any attempt
to *instantiate* one raises a clear ImportError instead of an AttributeError
from three frames deep.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from aegis_sql.observability.logging import get_logger

log = get_logger("training.model")

try:  # pragma: no cover - trivially exercised in any install that has torch
    import torch
    import torch.nn.functional as F
    from torch import Tensor, nn

    TORCH_AVAILABLE = True
    Module = nn.Module
except ImportError:  # pragma: no cover - torch-less installs

    class _MissingTorch:
        """Import-time stand-in so torch-less installs still get a readable error."""

        @staticmethod
        def no_grad() -> Any:
            def _decorator(fn: Any) -> Any:
                return fn

            return _decorator

        def __getattr__(self, name: str) -> Any:
            raise ImportError(
                "PyTorch is required for aegis_sql.training; install it with "
                "`pip install 'aegis-sql[train]'`."
            )

    torch = _MissingTorch()  # type: ignore[assignment]
    nn = torch  # type: ignore[assignment]
    F = torch  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    Module = object  # type: ignore[misc,assignment]
    TORCH_AVAILABLE = False


CONFIG_FILE = "config.json"
WEIGHTS_FILE = "model.pt"


def require_torch() -> None:
    """Fail fast, and in one place, when the training extra is not installed."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for aegis_sql.training; install it with "
            "`pip install 'aegis-sql[train]'`."
        )


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class AegisLMConfig:
    """Everything needed to rebuild the model byte-identically from disk."""

    vocab_size: int
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    max_seq_len: int = 512
    dropout: float = 0.1
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    norm_eps: float = 1e-5
    #: Free-form provenance (tokenizer hash, corpus size, git sha, ...).
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AegisLMConfig:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #


class RMSNorm(Module):
    """Root-mean-square layer norm (Zhang & Sennrich 2019): no mean, no bias."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # Accumulate in fp32 so the norm is stable even under autocast/bf16.
        dtype = x.dtype
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed.to(dtype)) * self.weight


class RotaryEmbedding(Module):
    """Precomputed RoPE cos/sin tables (Su et al. 2021)."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE requires an even head_dim")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)  # (T, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, head_dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, offset: int = 0) -> tuple[Tensor, Tensor]:
        end = offset + seq_len
        if end > self.max_seq_len:
            raise ValueError(f"sequence position {end} exceeds max_seq_len={self.max_seq_len}")
        cos = self.cos_cached[offset:end]
        sin = self.sin_cached[offset:end]
        return cos[None, None, :, :], sin[None, None, :, :]


def rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Rotate the query/key head vectors in place-equivalent fashion."""
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class Attention(Module):
    """Multi-head causal self-attention with an optional KV cache."""

    def __init__(self, cfg: AegisLMConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.dropout_p = cfg.dropout
        inner = cfg.n_heads * cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, inner, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, inner, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, inner, bias=False)
        self.o_proj = nn.Linear(inner, cfg.d_model, bias=False)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        attn_bias: Tensor | None = None,
        past_kv: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        bsz, seq_len, _ = x.shape
        shape = (bsz, seq_len, self.n_heads, self.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        present = (k, v) if use_cache else None

        # `is_causal` is only correct when queries and keys are aligned; with a
        # cache the causal structure is already encoded in the explicit bias.
        is_causal = attn_bias is None and past_kv is None and seq_len > 1
        drop = self.dropout_p if self.training else 0.0
        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_bias, dropout_p=drop, is_causal=is_causal
            )
        else:  # pragma: no cover - torch < 2.0 fallback
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if is_causal:
                mask = torch.ones(seq_len, k.shape[2], dtype=torch.bool, device=x.device).tril()
                scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            elif attn_bias is not None:
                scores = scores + attn_bias
            out = F.dropout(scores.softmax(-1), p=drop, training=self.training) @ v

        out = out.transpose(1, 2).reshape(bsz, seq_len, self.n_heads * self.head_dim)
        return self.resid_dropout(self.o_proj(out)), present


class SwiGLU(Module):
    """Gated feed-forward block: ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, cfg: AegisLMConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class Block(Module):
    """One pre-norm transformer layer."""

    def __init__(self, cfg: AegisLMConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        attn_bias: Tensor | None = None,
        past_kv: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        delta, present = self.attn(self.attn_norm(x), cos, sin, attn_bias, past_kv, use_cache)
        x = x + delta
        return x + self.mlp(self.mlp_norm(x)), present


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #


class AegisLM(Module):
    """A small causal language model that produces SQL from a Korean prompt."""

    def __init__(self, cfg: AegisLMConfig) -> None:
        require_torch()
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.rotary = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)

        self.apply(self._init_weights)
        # Scale the two residual-writing projections per GPT-2 §2.3, so the
        # variance of the residual stream does not grow with depth.
        residual_std = 0.02 / math.sqrt(2 * cfg.n_layers)
        for block in self.layers:
            nn.init.normal_(block.attn.o_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.down_proj.weight, mean=0.0, std=residual_std)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    @staticmethod
    def _init_weights(module: Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # -- forward ----------------------------------------------------------- #

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        past_key_values: list[tuple[Tensor, Tensor]] | None = None,
        use_cache: bool = False,
    ) -> dict[str, Any]:
        """Run the stack.

        ``labels`` follows the HuggingFace convention: same shape as
        ``input_ids``, shifted internally, ``-100`` where loss is suppressed.
        ``attention_mask`` is a ``(B, S)`` padding mask over *cached + current*
        positions, 1 for real tokens.
        """
        bsz, seq_len = input_ids.shape
        past_len = past_key_values[0][0].shape[2] if past_key_values else 0
        if past_len + seq_len > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence length {past_len + seq_len} exceeds max_seq_len={self.cfg.max_seq_len}"
            )

        x = self.dropout(self.embed_tokens(input_ids))
        cos, sin = self.rotary(seq_len, offset=past_len)
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)
        attn_bias = self._attention_bias(attention_mask, bsz, seq_len, past_len, x)

        presents: list[tuple[Tensor, Tensor]] = []
        for idx, block in enumerate(self.layers):
            past = past_key_values[idx] if past_key_values else None
            x, present = block(x, cos, sin, attn_bias, past, use_cache)
            if present is not None:
                presents.append(present)

        logits = self.lm_head(self.norm(x))
        out: dict[str, Any] = {"logits": logits, "past_key_values": presents if use_cache else None}

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            out["loss"] = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return out

    def _attention_bias(
        self,
        attention_mask: Tensor | None,
        bsz: int,
        seq_len: int,
        past_len: int,
        ref: Tensor,
    ) -> Tensor | None:
        """Build an additive ``(B, 1, T, S)`` mask, or None to use the fast causal path."""
        needs_causal = seq_len > 1
        if attention_mask is None and (past_len == 0 or not needs_causal):
            return None

        total = past_len + seq_len
        neg = torch.finfo(ref.dtype).min
        bias = torch.zeros(seq_len, total, dtype=ref.dtype, device=ref.device)
        if needs_causal:
            allowed = torch.ones(seq_len, total, dtype=torch.bool, device=ref.device).tril(past_len)
            bias = bias.masked_fill(~allowed, neg)
        bias = bias.expand(bsz, 1, seq_len, total).clone()
        if attention_mask is not None:
            pad = attention_mask[:, None, None, :total] == 0
            bias = bias.masked_fill(pad, neg)
        return bias

    # -- generation --------------------------------------------------------- #

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        eos_id: int | None = None,
        use_cache: bool = True,
    ) -> Tensor:
        """Autoregressive decoding.  ``temperature <= 0`` means greedy (deterministic)."""
        was_training = self.training
        self.eval()
        try:
            generated = input_ids
            past: list[tuple[Tensor, Tensor]] | None = None
            cursor = input_ids
            finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
            budget = min(max_new_tokens, self.cfg.max_seq_len - input_ids.shape[1])

            for _ in range(max(0, budget)):
                out = self.forward(cursor, past_key_values=past, use_cache=use_cache)
                logits = out["logits"][:, -1, :]
                past = out["past_key_values"] if use_cache else None

                if temperature and temperature > 0.0:
                    next_id = _sample(logits / temperature, top_p)
                else:
                    next_id = logits.argmax(dim=-1, keepdim=True)

                if eos_id is not None:
                    next_id = torch.where(finished[:, None], torch.full_like(next_id, eos_id), next_id)
                    finished = finished | (next_id.squeeze(-1) == eos_id)

                generated = torch.cat([generated, next_id], dim=1)
                cursor = next_id if use_cache else generated
                if eos_id is not None and bool(finished.all()):
                    break
            return generated
        finally:
            self.train(was_training)

    # -- housekeeping ------------------------------------------------------- #

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Parameter count; tied weights are counted once (``parameters()`` dedupes)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def save_pretrained(self, path: str | Path) -> Path:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        (out / CONFIG_FILE).write_text(
            json.dumps(self.cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        state = dict(self.state_dict())
        if self.cfg.tie_embeddings:
            state.pop("lm_head.weight", None)  # re-tied on load; do not store twice
        torch.save(state, out / WEIGHTS_FILE)
        log.info("checkpoint saved", path=str(out), params=self.num_parameters())
        return out

    @classmethod
    def from_pretrained(cls, path: str | Path, map_location: str = "cpu") -> AegisLM:
        require_torch()
        src = Path(path)
        cfg = AegisLMConfig.from_dict(json.loads((src / CONFIG_FILE).read_text(encoding="utf-8")))
        model = cls(cfg)
        state = torch.load(src / WEIGHTS_FILE, map_location=map_location, weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        allowed = {"lm_head.weight"} if cfg.tie_embeddings else set()
        leftover = [k for k in missing if k not in allowed]
        if leftover or unexpected:
            log.warning("checkpoint key mismatch", missing=leftover[:5], unexpected=list(unexpected)[:5])
        if cfg.tie_embeddings:
            model.lm_head.weight = model.embed_tokens.weight
        model.eval()
        return model


def _sample(logits: Tensor, top_p: float) -> Tensor:
    """Nucleus sampling; ``top_p >= 1`` degenerates to plain multinomial sampling."""
    probs = logits.softmax(dim=-1)
    if top_p < 1.0:
        ordered, index = probs.sort(dim=-1, descending=True)
        cumulative = ordered.cumsum(dim=-1)
        # Keep the first token that crosses the threshold, drop everything after.
        drop = cumulative - ordered > top_p
        ordered = ordered.masked_fill(drop, 0.0)
        probs = torch.zeros_like(probs).scatter(1, index, ordered)
        probs = probs / probs.sum(dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)
