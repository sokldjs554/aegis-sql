"""Supervised fine-tuning: the stage that turns a language model into a SQL writer.

Three details in here are what actually decide whether the sLLM tier is usable,
and all three are easy to get quietly wrong:

**Loss masking.**  Labels are ``-100`` across the entire prompt, so the model is
never rewarded for reproducing the schema card.  Without this the prompt (which
is 10-40x longer than the SQL) dominates the objective and the model learns to
be a very good schema-card autocompleter that emits mediocre SQL.  The boundary
is the ``<|sql|>`` token, a real vocabulary item rather than a string match, so
the split is unambiguous even when the SQL itself contains the word "SQL".

**Left truncation.**  When prompt + SQL exceed the context, the *head of the
schema card* is dropped, never the tail.  The question sits at the end of the
prompt and the retrieval layer emits the most relevant tables first-to-last in
descending relevance, so right truncation would delete the question — the one
span the model cannot do without.  This is also why
:func:`build_prompt` fixes the schema-then-question order permanently: change it
and every checkpoint trained before the change silently degrades.

**Prompt stability.**  :func:`build_prompt` is the single definition of the SLM
prompt, shared verbatim by training (:class:`SQLDataset`) and serving
(:mod:`aegis_sql.training.infer`).  A 15M-parameter model has no capacity to
spare on format robustness; drifting the template by one newline between train
and serve is worth several points of execution accuracy.

The optimiser recipe is deliberately unexciting and reproducible: AdamW with
``betas=(0.9, 0.95)`` (the low-noise setting the GPT-3/Llama recipes use for
small batches), decoupled weight decay applied only to matrices, linear warmup
into cosine decay, gradient accumulation to emulate a batch that will not fit,
and clipping at 1.0.  Everything is seeded, and mixed precision is gated behind
an actual CUDA device so a CPU run stays bit-reproducible.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aegis_sql.config import TrainingConfig
from aegis_sql.observability.logging import get_logger
from aegis_sql.training.model import TORCH_AVAILABLE, AegisLM, Tensor, require_torch, torch
from aegis_sql.training.tokenizer import ByteBPETokenizer

log = get_logger("training.sft")

if TORCH_AVAILABLE:  # pragma: no cover - trivially true wherever training runs
    from torch.utils.data import DataLoader
    from torch.utils.data import Dataset as _TorchDataset
else:  # pragma: no cover - torch-less installs still import this module
    _TorchDataset = object  # type: ignore[assignment,misc]
    DataLoader = None  # type: ignore[assignment]

IGNORE_INDEX = -100

#: THE canonical sLLM prompt.  Schema first, question last (see module docstring).
PROMPT_TEMPLATE = "### 스키마\n{schema_card}\n### 질문\n{question}"


def build_prompt(question: str, schema_card: str) -> str:
    """Render the fixed SLM prompt.

    The ``<|sql|>`` separator and the generated SQL are *not* part of this
    string: the caller appends the separator as a token id so that training and
    decoding agree on the boundary without any string parsing.
    """
    return PROMPT_TEMPLATE.format(schema_card=schema_card.strip(), question=question.strip())


@dataclass(slots=True)
class SFTExample:
    """One (prompt, SQL) supervision pair.  ``prompt`` is already rendered."""

    prompt: str
    target: str

    @classmethod
    def from_question(cls, question: str, sql: str, schema_card: str = "") -> SFTExample:
        return cls(prompt=build_prompt(question, schema_card), target=sql.strip())


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


class SQLDataset(_TorchDataset):
    """Tokenises ``prompt <|sql|> target <|eos|>`` and masks the loss to the target."""

    def __init__(
        self,
        examples: Sequence[SFTExample],
        tokenizer: ByteBPETokenizer,
        max_seq_len: int = 512,
        min_prompt_tokens: int = 16,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.min_prompt_tokens = min_prompt_tokens
        self.features: list[dict[str, list[int]]] = [self._encode(ex) for ex in examples]
        self.n_truncated = sum(1 for f in self.features if f["truncated"][0])
        if self.n_truncated:
            log.info(
                "SFT examples truncated",
                n=self.n_truncated,
                total=len(self.features),
                max_seq_len=max_seq_len,
            )

    def _encode(self, example: SFTExample) -> dict[str, list[int]]:
        tok = self.tokenizer
        prompt_ids = tok.encode(example.prompt, add_bos=True)
        target_ids = tok.encode(example.target) + [tok.eos_id]
        truncated = False

        # Reserve room for the separator and the whole target; the target is only
        # clipped when it alone cannot fit, in which case the EOS is preserved.
        room = self.max_seq_len - 1
        if len(target_ids) > room - self.min_prompt_tokens:
            target_ids = target_ids[: max(1, room - self.min_prompt_tokens - 1)] + [tok.eos_id]
            truncated = True
        budget = room - len(target_ids)
        if len(prompt_ids) > budget:
            prompt_ids = [tok.bos_id] + prompt_ids[1:][-(budget - 1) :]
            truncated = True

        input_ids = prompt_ids + [tok.sep_id] + target_ids
        labels = [IGNORE_INDEX] * (len(prompt_ids) + 1) + target_ids
        return {"input_ids": input_ids, "labels": labels, "truncated": [int(truncated)]}

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        item = self.features[index]
        return {"input_ids": item["input_ids"], "labels": item["labels"]}

    @property
    def token_stats(self) -> dict[str, float]:
        lengths = [len(f["input_ids"]) for f in self.features]
        if not lengths:
            return {"n": 0, "mean": 0.0, "max": 0, "p95": 0}
        ordered = sorted(lengths)
        return {
            "n": len(lengths),
            "mean": round(sum(lengths) / len(lengths), 1),
            "max": ordered[-1],
            "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        }


def collate(batch: list[dict[str, list[int]]], pad_id: int) -> dict[str, Tensor]:
    """Right-pad a batch and build the padding mask (1 = real token)."""
    require_torch()
    width = max(len(item["input_ids"]) for item in batch)
    input_ids, labels, attention = [], [], []
    for item in batch:
        pad = width - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_id] * pad)
        labels.append(item["labels"] + [IGNORE_INDEX] * pad)
        attention.append([1] * len(item["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
    }


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #


class SFTTrainer:
    """AdamW + warmup/cosine SFT loop that runs to completion on four CPU cores."""

    def __init__(
        self,
        model: AegisLM,
        tokenizer: ByteBPETokenizer,
        cfg: TrainingConfig,
        device: str = "auto",
        log_every: int = 20,
    ) -> None:
        require_torch()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.device = resolve_device(device if device != "auto" else cfg.device)
        self.model = model.to(self.device)
        self.log_every = max(1, log_every)
        self.use_amp = self.device.type == "cuda"

    # -- public ------------------------------------------------------------ #

    def train(
        self,
        train_examples: Sequence[SFTExample],
        dev_examples: Sequence[SFTExample] | None = None,
        output_dir: str | Path | None = None,
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        """Fine-tune on ``train_examples``; returns the training history."""
        if not train_examples:
            raise ValueError("SFTTrainer.train received an empty training set")
        seed_everything(self.cfg.seed)

        train_ds = SQLDataset(train_examples, self.tokenizer, self.cfg.max_seq_len)
        dev_ds = SQLDataset(dev_examples, self.tokenizer, self.cfg.max_seq_len) if dev_examples else None
        loader = self._loader(train_ds, shuffle=True)
        dev_loader = self._loader(dev_ds, shuffle=False) if dev_ds else None

        accum = max(1, self.cfg.grad_accum)
        steps_per_epoch = max(1, math.ceil(len(loader) / accum))
        total_steps = steps_per_epoch * max(1, self.cfg.epochs)
        if max_steps is not None:
            total_steps = min(total_steps, max_steps)

        optimizer = self._build_optimizer()
        warmup = max(1, int(total_steps * self.cfg.warmup_ratio))
        scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        history: dict[str, Any] = {
            "train_loss": [],
            "dev_loss": [],
            "dev_token_acc": [],
            "steps": 0,
            "wall_s": 0.0,
        }
        best_dev = math.inf
        best_state: dict[str, Any] | None = None
        started = time.perf_counter()
        step = 0
        stop = False

        log.info(
            "SFT start",
            examples=len(train_ds),
            dev=len(dev_ds) if dev_ds else 0,
            total_steps=total_steps,
            device=str(self.device),
            params=self.model.num_parameters(trainable_only=True),
            tokens=train_ds.token_stats,
        )

        for epoch in range(max(1, self.cfg.epochs)):
            self.model.train()
            running, seen = 0.0, 0
            optimizer.zero_grad(set_to_none=True)

            for micro, batch in enumerate(loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss = self._forward_loss(batch, scaler)
                running += loss * batch["input_ids"].shape[0]
                seen += batch["input_ids"].shape[0]

                is_boundary = (micro + 1) % accum == 0 or micro + 1 == len(loader)
                if not is_boundary:
                    continue

                self._optimizer_step(optimizer, scaler, self._lr_at(step, warmup, total_steps))
                step += 1
                history["steps"] = step
                if step % self.log_every == 0 or step == total_steps:
                    log.info(
                        "step",
                        step=step,
                        total=total_steps,
                        epoch=epoch + 1,
                        loss=round(running / max(1, seen), 4),
                        lr=round(self._lr_at(step, warmup, total_steps), 6),
                    )
                if step >= total_steps:
                    stop = True
                    break

            epoch_loss = running / max(1, seen)
            history["train_loss"].append(round(epoch_loss, 5))

            if dev_loader is not None:
                dev_loss, dev_acc = self.evaluate(dev_loader)
                history["dev_loss"].append(round(dev_loss, 5))
                history["dev_token_acc"].append(round(dev_acc, 5))
                log.info("epoch", epoch=epoch + 1, train_loss=round(epoch_loss, 4),
                         dev_loss=round(dev_loss, 4), dev_token_acc=round(dev_acc, 4))
                score = dev_loss
            else:
                log.info("epoch", epoch=epoch + 1, train_loss=round(epoch_loss, 4))
                score = epoch_loss

            if score < best_dev:
                best_dev = score
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                if output_dir is not None:
                    self.save(output_dir)
            if stop:
                break

        # Restore the best epoch so the caller always holds the checkpoint it saved.
        if best_state is not None:
            self.model.load_state_dict(best_state)
        history["wall_s"] = round(time.perf_counter() - started, 2)
        history["best_score"] = round(best_dev, 5) if math.isfinite(best_dev) else None
        log.info("SFT done", steps=history["steps"], wall_s=history["wall_s"], best=history["best_score"])
        return history

    @torch.no_grad()
    def evaluate(self, loader: Iterable[dict[str, Tensor]]) -> tuple[float, float]:
        """Mean token cross-entropy and next-token accuracy over the target span."""
        was_training = self.model.training
        self.model.eval()
        total_loss, total_tokens, correct = 0.0, 0, 0
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            logits = out["logits"][:, :-1, :]
            labels = batch["labels"][:, 1:]
            mask = labels != IGNORE_INDEX
            n = int(mask.sum())
            if n == 0:
                continue
            total_loss += float(out["loss"]) * n
            total_tokens += n
            correct += int((logits.argmax(-1)[mask] == labels[mask]).sum())
        self.model.train(was_training)
        if total_tokens == 0:
            return math.inf, 0.0
        return total_loss / total_tokens, correct / total_tokens

    def save(self, output_dir: str | Path) -> Path:
        """Persist a servable checkpoint plus the tokenizer.

        Under LoRA the small adapter is written next to a *merged* full
        checkpoint, so the directory is loadable by
        :meth:`AegisLM.from_pretrained` while training continues unmerged.
        """
        from aegis_sql.training.lora import WEIGHTS_MERGE_NOTE, lora_modules, merged_state_dict, save_lora

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        if lora_modules(self.model):
            save_lora(self.model, out)
            state = merged_state_dict(self.model)
            if self.model.cfg.tie_embeddings:
                state.pop("lm_head.weight", None)
            torch.save(state, out / "model.pt")
            (out / "config.json").write_text(
                json.dumps(self.model.cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (out / "README.txt").write_text(WEIGHTS_MERGE_NOTE, encoding="utf-8")
        else:
            self.model.save_pretrained(out)
        self.tokenizer.save(out)
        return out

    # -- internals ---------------------------------------------------------- #

    def _loader(self, dataset: SQLDataset | None, shuffle: bool) -> Any:
        if dataset is None:
            return None
        generator = torch.Generator()
        generator.manual_seed(self.cfg.seed)
        return DataLoader(
            dataset,
            batch_size=max(1, self.cfg.batch_size),
            shuffle=shuffle,
            generator=generator if shuffle else None,
            num_workers=0,
            collate_fn=lambda b: collate(b, self.tokenizer.pad_id),
        )

    def _build_optimizer(self) -> Any:
        """Decay matrices, never norms/biases — the standard decoupled-decay split."""
        decay, no_decay = [], []
        for param in self.model.parameters():
            if not param.requires_grad:
                continue
            (decay if param.dim() >= 2 else no_decay).append(param)
        groups = [
            {"params": decay, "weight_decay": self.cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=self.cfg.lr, betas=(0.9, 0.95), eps=1e-8)

    def _lr_at(self, step: int, warmup: int, total: int) -> float:
        if step < warmup:
            return self.cfg.lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        # Cosine decay to 10% of peak: a hard zero wastes the last epoch.
        return self.cfg.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

    def _forward_loss(self, batch: dict[str, Tensor], scaler: Any) -> float:
        accum = max(1, self.cfg.grad_accum)
        if self.use_amp:  # pragma: no cover - CPU CI never takes this branch
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
            scaler.scale(out["loss"] / accum).backward()
        else:
            out = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            (out["loss"] / accum).backward()
        return float(out["loss"].detach())

    def _optimizer_step(self, optimizer: Any, scaler: Any, lr: float) -> None:
        for group in optimizer.param_groups:
            group["lr"] = lr
        if scaler is not None:  # pragma: no cover - CUDA only
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def seed_everything(seed: int) -> None:
    """Seed python/torch (and numpy when present) — training must be replayable."""
    random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - CUDA only
            torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:  # pragma: no cover - numpy is a hard dependency in practice
        pass


def resolve_device(device: str) -> Any:
    require_torch()
    if device in {"auto", ""}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
