"""LoRA adapters, implemented rather than imported.

The engine has to be re-fitted continuously — the flywheel emits a new corpus
every time the schema changes or a repair loop discovers a systematic mistake —
and full fine-tuning is the wrong tool for that cadence: it produces a 60MB
checkpoint per experiment, needs the optimiser state for every weight, and makes
"ship the delta for the 손해보험 schema on top of the shared base" impossible.
Low-rank adapters (Hu et al. 2021) reduce a re-fit to a few hundred KB of
``A``/``B`` matrices that can be stored per tenant, diffed, A/B tested, and
merged into the base weights at deploy time so inference pays exactly zero
overhead.

Two properties are load-bearing and are asserted by the test-suite:

* **``B`` is initialised to zero**, so ``W + (alpha/r)·B·A == W`` at step 0.  A
  freshly adapted model is *bit-identical* to the model it wraps — the adapter
  can never silently degrade a validated checkpoint before a single gradient
  step has been taken.
* **Merging is exact and reversible.**  :meth:`LoRALinear.merge` folds the delta
  into the frozen base weight for deployment, :meth:`LoRALinear.unmerge` takes
  it back out for further training, and :func:`merge_lora_` replaces the wrapper
  modules with plain ``nn.Linear`` so a merged checkpoint has no LoRA-shaped
  keys and loads into the vanilla architecture.

Targeting is by *attribute name* (``q_proj``, ``v_proj``, ``o_proj``, ...), which
is why :mod:`aegis_sql.training.model` names its projections the way the
Llama family does.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from aegis_sql.observability.logging import get_logger
from aegis_sql.training.model import TORCH_AVAILABLE, Module, Tensor, nn, require_torch, torch

log = get_logger("training.lora")

LORA_FILE = "adapter.pt"

#: Dropped next to a LoRA checkpoint so nobody has to guess what the two files are.
WEIGHTS_MERGE_NOTE = (
    "model.pt  : base weights with the LoRA adapter already merged in (servable as-is)\n"
    "adapter.pt: the adapter alone — reload it onto the *unmerged* base to keep training\n"
)


class LoRALinear(Module):
    """Wraps a frozen ``nn.Linear`` with a trainable rank-``r`` residual branch."""

    def __init__(self, base: Any, r: int = 8, alpha: int = 16, dropout: float = 0.0) -> None:
        require_torch()
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.merged = False

        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        # Kaiming on A, zeros on B: the product starts at exactly zero.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def in_features(self) -> int:
        return int(self.base.in_features)

    @property
    def out_features(self) -> int:
        return int(self.base.out_features)

    def forward(self, x: Tensor) -> Tensor:
        out = self.base(x)
        if self.merged:
            return out
        delta = self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)
        return out + delta * self.scaling

    def delta_weight(self) -> Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling

    @torch.no_grad()
    def merge(self) -> None:
        """Fold the adapter into the base weight (idempotent)."""
        if not self.merged:
            self.base.weight.data += self.delta_weight().to(self.base.weight.dtype)
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if self.merged:
            self.base.weight.data -= self.delta_weight().to(self.base.weight.dtype)
            self.merged = False

    def extra_repr(self) -> str:  # pragma: no cover - display helper
        return f"r={self.r}, alpha={self.alpha}, merged={self.merged}"


# --------------------------------------------------------------------------- #
# Model surgery
# --------------------------------------------------------------------------- #


def apply_lora(
    model: Any,
    target_names: list[str],
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
) -> int:
    """Wrap every ``nn.Linear`` child whose attribute name is in ``target_names``.

    Returns the number of modules wrapped.  Already-wrapped modules are skipped,
    so calling this twice is safe.
    """
    require_torch()
    wanted = set(target_names)
    wrapped = 0
    for _, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name not in wanted or isinstance(child, LoRALinear):
                continue
            if not isinstance(child, nn.Linear):
                continue
            setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
            wrapped += 1
    log.info("LoRA applied", wrapped=wrapped, targets=sorted(wanted), r=r, alpha=alpha)
    return wrapped


def mark_only_lora_trainable(model: Any) -> tuple[int, int]:
    """Freeze everything except ``lora_A``/``lora_B``.  Returns ``(trainable, total)``."""
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        is_lora = "lora_A" in name or "lora_B" in name
        param.requires_grad_(is_lora)
        total += param.numel()
        if is_lora:
            trainable += param.numel()
    log.info(
        "LoRA parameters marked",
        trainable=trainable,
        total=total,
        pct=round(100.0 * trainable / max(1, total), 3),
    )
    return trainable, total


def lora_modules(model: Any) -> list[tuple[str, LoRALinear]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]


def lora_state_dict(model: Any) -> dict[str, Any]:
    """Only the adapter tensors — a few hundred KB instead of a whole checkpoint."""
    return {
        name: param.detach().cpu().clone()
        for name, param in model.state_dict().items()
        if "lora_A" in name or "lora_B" in name
    }


def load_lora_state_dict(model: Any, state: dict[str, Any]) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise KeyError(f"adapter contains keys the model does not have: {sorted(unexpected)[:5]}")
    adapter_keys = {k for k in missing if "lora_A" in k or "lora_B" in k}
    if adapter_keys:
        log.warning("adapter did not cover every LoRA slot", missing=sorted(adapter_keys)[:5])


def save_lora(model: Any, path: str | Path) -> Path:
    target = _resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state_dict(model), target)
    log.info("adapter saved", path=str(target))
    return target


def load_lora(model: Any, path: str | Path) -> None:
    require_torch()
    target = _resolve_path(path)
    state = torch.load(target, map_location="cpu", weights_only=True)
    load_lora_state_dict(model, state)


@torch.no_grad()
def merged_state_dict(model: Any) -> dict[str, Any]:
    """A *vanilla-architecture* state dict with the adapters folded in.

    Merges, rewrites ``…q_proj.base.weight`` back to ``…q_proj.weight``, drops
    the adapter tensors, then unmerges — so an adapted model can be checkpointed
    mid-training in a form :meth:`AegisLM.from_pretrained` loads directly,
    without ending the training run.
    """
    modules = lora_modules(model)
    for _, module in modules:
        module.merge()
    try:
        state: dict[str, Any] = {}
        for key, value in model.state_dict().items():
            if "lora_A" in key or "lora_B" in key:
                continue
            state[key.replace(".base.", ".")] = value.detach().cpu().clone()
        return state
    finally:
        for _, module in modules:
            module.unmerge()


def merge_lora_(model: Any) -> None:
    """Fold every adapter in and swap the wrappers out for plain ``nn.Linear``.

    After this the model has no LoRA parameters at all, so
    :meth:`AegisLM.save_pretrained` writes a checkpoint that loads into the
    unadapted architecture — which is what production serves.
    """
    for _, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                child.merge()
                child.base.weight.requires_grad_(True)
                if child.base.bias is not None:
                    child.base.bias.requires_grad_(True)
                setattr(module, child_name, child.base)


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_dir() or not p.suffix:
        return p / LORA_FILE
    return p


__all__ = [
    "LORA_FILE",
    "TORCH_AVAILABLE",
    "WEIGHTS_MERGE_NOTE",
    "LoRALinear",
    "apply_lora",
    "load_lora",
    "load_lora_state_dict",
    "lora_modules",
    "lora_state_dict",
    "mark_only_lora_trainable",
    "merge_lora_",
    "merged_state_dict",
    "save_lora",
]
