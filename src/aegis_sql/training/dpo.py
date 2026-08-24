"""Direct Preference Optimization on the engine's own failures.

SFT can only teach the model what a correct query looks like.  It cannot teach
it what *nearly* correct looks like, and nearly-correct is exactly what a small
model produces: the right tables, the right aggregate, and ``CTRT_STAT_CD =
'01'`` where the question meant ``'02'``.  Those mistakes are systematic, and
the engine already records them for free — every entry in the repair loop's
history is a pair ``(what the model wrote, what actually executed and was
verified)``.  :func:`pairs_from_repair_log` turns that log into preference data,
which closes the flywheel: production failures become the next training signal
without anyone labelling anything.

DPO (Rafailov et al. 2023) is the right fit rather than PPO because there is no
reward model to fit and no rollout loop to stabilise — the preference pair *is*
the reward, and the KL anchor is a frozen copy of the SFT checkpoint.  The
implementation here is the objective as written in the paper, with two details
that matter in practice:

* Sequence log-probabilities are summed **over the target span only** (labels
  masked to ``-100`` on the prompt, exactly as in
  :mod:`aegis_sql.training.sft`).  Including the prompt would make the implicit
  reward dominated by how surprising the schema card is, which is identical for
  both branches and therefore pure variance.
* The **implicit reward margin** ``beta·((logπ_c - logπ_ref_c) - (logπ_r -
  logπ_ref_r))`` is tracked every step alongside the fraction of pairs where it
  is positive.  Loss going down while the margin stays flat is the classic sign
  that the reference model is not actually frozen — worth being able to see.

The learning rate defaults to a tenth of the SFT rate: preference optimisation
on a small model diverges into degenerate low-entropy policies at SFT rates.
"""

from __future__ import annotations

import copy
import math
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aegis_sql.config import TrainingConfig
from aegis_sql.observability.logging import get_logger
from aegis_sql.training.model import F, Tensor, require_torch, torch
from aegis_sql.training.sft import (
    IGNORE_INDEX,
    SFTExample,
    SQLDataset,
    build_prompt,
    collate,
    resolve_device,
    seed_everything,
)
from aegis_sql.training.tokenizer import ByteBPETokenizer

log = get_logger("training.dpo")


@dataclass(slots=True)
class DPOExample:
    """One preference triple.  ``prompt`` must be a rendered SLM prompt."""

    prompt: str
    chosen: str
    rejected: str

    @classmethod
    def from_question(cls, question: str, gold_sql: str, failed_sql: str, schema_card: str = "") -> DPOExample:
        return cls(build_prompt(question, schema_card), gold_sql.strip(), failed_sql.strip())


def pairs_from_repair_log(records: Sequence[dict[str, Any]]) -> list[DPOExample]:
    """Build ``(gold, failed)`` preference pairs from the repair loop's history.

    Expects ``{"prompt": ..., "gold_sql": ..., "failed_sql": ...}`` records and
    silently drops the useless ones: missing fields, and pairs whose two SQL
    strings are equal up to whitespace and case (a repair that only reformatted
    teaches nothing and contributes a zero-margin gradient).
    """
    pairs: list[DPOExample] = []
    for record in records:
        prompt = str(record.get("prompt") or "").strip()
        gold = str(record.get("gold_sql") or "").strip()
        failed = str(record.get("failed_sql") or "").strip()
        if not prompt or not gold or not failed:
            continue
        if " ".join(gold.lower().split()) == " ".join(failed.lower().split()):
            continue
        pairs.append(DPOExample(prompt=prompt, chosen=gold, rejected=failed))
    log.info("preference pairs built", kept=len(pairs), seen=len(records))
    return pairs


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #


class DPOTrainer:
    """Preference-optimises a policy against a frozen reference copy."""

    def __init__(
        self,
        policy_model: Any,
        ref_model: Any | None,
        tokenizer: ByteBPETokenizer,
        cfg: TrainingConfig,
        beta: float = 0.1,
        device: str = "auto",
        lr: float | None = None,
        log_every: int = 5,
    ) -> None:
        require_torch()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.beta = beta if beta is not None else cfg.dpo_beta
        self.device = resolve_device(device if device != "auto" else cfg.device)
        self.policy = policy_model.to(self.device)
        self.lr = lr if lr is not None else cfg.lr * 0.1
        self.log_every = max(1, log_every)

        # No reference given → snapshot the policy now.  That is the correct
        # anchor: the KL term must point at the SFT checkpoint, not at init.
        reference = ref_model if ref_model is not None else copy.deepcopy(policy_model)
        self.ref = reference.to(self.device)
        self.ref.eval()
        for param in self.ref.parameters():
            param.requires_grad_(False)

    # -- public ------------------------------------------------------------ #

    def train(
        self,
        examples: Sequence[DPOExample],
        output_dir: str | Path | None = None,
        max_steps: int | None = None,
        epochs: int | None = None,
    ) -> dict[str, Any]:
        """Run DPO over ``examples``; returns loss / margin / accuracy history."""
        if not examples:
            raise ValueError("DPOTrainer.train received no preference pairs")
        seed_everything(self.cfg.seed)

        chosen_ds = SQLDataset(
            [SFTExample(e.prompt, e.chosen) for e in examples], self.tokenizer, self.cfg.max_seq_len
        )
        rejected_ds = SQLDataset(
            [SFTExample(e.prompt, e.rejected) for e in examples], self.tokenizer, self.cfg.max_seq_len
        )

        batch_size = max(1, min(self.cfg.batch_size, len(examples)))
        n_epochs = epochs if epochs is not None else max(1, self.cfg.epochs)
        optimizer = torch.optim.AdamW(
            [p for p in self.policy.parameters() if p.requires_grad],
            lr=self.lr,
            betas=(0.9, 0.95),
            weight_decay=self.cfg.weight_decay,
        )

        history: dict[str, Any] = {"loss": [], "margin": [], "acc": [], "steps": 0, "wall_s": 0.0}
        started = time.perf_counter()
        step = 0
        order = list(range(len(examples)))
        rng = random.Random(self.cfg.seed)
        stop = False

        log.info(
            "DPO start",
            pairs=len(examples),
            beta=self.beta,
            lr=self.lr,
            batch_size=batch_size,
            device=str(self.device),
        )

        for _epoch in range(n_epochs):
            rng.shuffle(order)
            for start in range(0, len(order), batch_size):
                index = order[start : start + batch_size]
                stats = self._step(chosen_ds, rejected_ds, index, optimizer)
                step += 1
                history["loss"].append(round(stats["loss"], 5))
                history["margin"].append(round(stats["margin"], 5))
                history["acc"].append(round(stats["acc"], 5))
                if step % self.log_every == 0:
                    log.info("dpo step", step=step, **{k: round(v, 4) for k, v in stats.items()})
                if max_steps is not None and step >= max_steps:
                    stop = True
                    break
            if stop:
                break

        history["steps"] = step
        history["wall_s"] = round(time.perf_counter() - started, 2)
        window = max(1, step // 4)
        history["margin_start"] = round(sum(history["margin"][:window]) / window, 5)
        history["margin_end"] = round(sum(history["margin"][-window:]) / window, 5)
        history["final_acc"] = round(sum(history["acc"][-window:]) / window, 5)

        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            self.policy.save_pretrained(out)
            self.tokenizer.save(out)
        log.info(
            "DPO done",
            steps=step,
            wall_s=history["wall_s"],
            margin_start=history["margin_start"],
            margin_end=history["margin_end"],
            final_acc=history["final_acc"],
        )
        return history

    # -- internals ---------------------------------------------------------- #

    def _step(
        self,
        chosen_ds: SQLDataset,
        rejected_ds: SQLDataset,
        index: list[int],
        optimizer: Any,
    ) -> dict[str, float]:
        pad = self.tokenizer.pad_id
        chosen = self._to_device(collate([chosen_ds[i] for i in index], pad))
        rejected = self._to_device(collate([rejected_ds[i] for i in index], pad))

        self.policy.train()
        policy_chosen = sequence_logprobs(self.policy, chosen)
        policy_rejected = sequence_logprobs(self.policy, rejected)
        with torch.no_grad():
            ref_chosen = sequence_logprobs(self.ref, chosen)
            ref_rejected = sequence_logprobs(self.ref, rejected)

        chosen_reward = self.beta * (policy_chosen - ref_chosen)
        rejected_reward = self.beta * (policy_rejected - ref_rejected)
        loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            margin = float((chosen_reward - rejected_reward).mean())
            acc = float((chosen_reward > rejected_reward).float().mean())
        return {
            "loss": float(loss.detach()),
            "margin": margin,
            "acc": acc,
            "chosen_reward": float(chosen_reward.mean()),
            "rejected_reward": float(rejected_reward.mean()),
        }

    def _to_device(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return {k: v.to(self.device) for k, v in batch.items()}


def sequence_logprobs(model: Any, batch: dict[str, Tensor]) -> Tensor:
    """Sum of token log-probabilities over the *target span* of each sequence.

    Positions labelled ``-100`` (the prompt and any right padding) contribute
    nothing, so the returned ``(B,)`` tensor is comparable across examples whose
    prompts differ wildly in length.
    """
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out["logits"][:, :-1, :]
    labels = batch["labels"][:, 1:]
    mask = labels != IGNORE_INDEX
    gathered = logits.log_softmax(-1).gather(-1, labels.masked_fill(~mask, 0).unsqueeze(-1))
    return (gathered.squeeze(-1) * mask).sum(-1)


def reward_accuracy(margins: Sequence[float]) -> float:
    """Fraction of steps whose implicit reward preferred the chosen completion."""
    if not margins:
        return 0.0
    return sum(1 for m in margins if m > 0) / len(margins)


def is_improving(history: dict[str, Any]) -> bool:
    """Cheap sanity gate for CI: did the margin actually move in the right direction?"""
    start = history.get("margin_start")
    end = history.get("margin_end")
    return bool(start is not None and end is not None and math.isfinite(end) and end > start)
