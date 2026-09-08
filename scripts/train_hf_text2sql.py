#!/usr/bin/env python3
"""Fine-tune a pretrained Qwen Text-to-SQL baseline on the AEGIS flywheel.

This is the LitE-SQL-inspired comparison arm.  It deliberately reuses the exact
AEGIS train/dev split and schema-card renderer so the comparison against the
5.3M from-scratch AegisLM changes the pretrained model, not the data definition.

Install the optional stack explicitly; it is not part of normal CI:

    pip install -e ".[train,hf]"
    python scripts/train_hf_text2sql.py --qlora
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from aegis_sql.config import PROJECT_ROOT, get_settings
from aegis_sql.schema.card import SchemaCardBuilder
from aegis_sql.schema.graph import JoinGraph
from aegis_sql.schema.introspect import introspect
from aegis_sql.schema.profile import Profiler
from aegis_sql.training.hf_experiment import (
    SYSTEM_PROMPT,
    PreparedExample,
    experiment_manifest,
    load_jsonl,
    prepare_records,
    user_prompt,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


class TokenizedSQLDataset:
    """Prompt-masked causal-LM examples; only target SQL contributes to loss."""

    def __init__(self, examples, tokenizer, max_length: int) -> None:
        self.features = [self._encode(ex, tokenizer, max_length) for ex in examples]

    @staticmethod
    def _encode(example: PreparedExample, tokenizer, max_length: int) -> dict[str, list[int]]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(example)},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(example.sql, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]

        if len(target_ids) >= max_length:
            target_ids = target_ids[: max_length - 1] + [tokenizer.eos_token_id]
        prompt_budget = max(1, max_length - len(target_ids))
        if len(prompt_ids) > prompt_budget:
            prompt_ids = prompt_ids[-prompt_budget:]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.features[index]


class SQLCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        import torch

        width = max(len(item["input_ids"]) for item in batch)
        ids, masks, labels = [], [], []
        for item in batch:
            pad = width - len(item["input_ids"])
            ids.append(item["input_ids"] + [self.pad_token_id] * pad)
            masks.append(item["attention_mask"] + [0] * pad)
            labels.append(item["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _card_builder():
    settings = get_settings()
    db = _resolve(settings.database.path)
    if not db.exists():
        raise FileNotFoundError(f"demo database missing: {db}; run `make demo-db` first")
    schema = introspect(db)
    profile = Profiler(db, sample=settings.database.profile_sample).profile(schema)
    return schema, SchemaCardBuilder(schema, profile, JoinGraph(schema))


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _load_examples(data_dir: Path, split: str, builder, limit: int | None):
    path = data_dir / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"flywheel split missing: {path}; run `make flywheel` first")
    return path, prepare_records(load_jsonl(path, limit=limit), builder)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="LoRA/QLoRA Qwen baseline for AEGIS Text-to-SQL")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--data-dir", default="data/generated/flywheel")
    ap.add_argument("--out", default="data/generated/hf/qwen2.5-coder-1.5b")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-dev", type=int, default=None)
    ap.add_argument("--qlora", action="store_true", help="load base weights in 4-bit (CUDA only)")
    ap.add_argument(
        "--prepare-only",
        action="store_true",
        help="verify data/schema wiring and write manifest without importing HuggingFace",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = _resolve(args.data_dir)
    out = _resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)

    schema, builder = _card_builder()
    train_path, train_examples = _load_examples(data_dir, "train", builder, args.limit_train)
    dev_path, dev_examples = _load_examples(data_dir, "dev", builder, args.limit_dev)
    manifest = experiment_manifest(
        model=args.model,
        train_path=train_path,
        dev_path=dev_path,
        train_count=len(train_examples),
        dev_count=len(dev_examples),
        seed=args.seed,
        schema_fingerprint=schema.fingerprint(),
        qlora=args.qlora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    manifest["git_sha"] = _git_sha()
    manifest["max_length"] = args.max_length
    manifest["status"] = "prepared"
    manifest_path = out / "experiment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"prepared: train={len(train_examples)} dev={len(dev_examples)} schema={schema.fingerprint()}")
    print(f"manifest: {manifest_path}")

    if args.prepare_only:
        return 0

    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit('HuggingFace experiment dependencies missing; run `pip install -e ".[train,hf]"`') from exc

    if args.qlora and not torch.cuda.is_available():
        raise SystemExit("--qlora requires CUDA; omit it for a CPU/GPU full-precision LoRA run")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization = None
    if args.qlora:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        quantization_config=quantization,
        device_map="auto" if args.qlora else None,
    )
    if args.qlora:
        model = prepare_model_for_kbit_training(model)
    elif torch.cuda.is_available():
        model = model.cuda()

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        ),
    )
    model.print_trainable_parameters()

    train_ds = TokenizedSQLDataset(train_examples, tokenizer, args.max_length)
    dev_ds = TokenizedSQLDataset(dev_examples, tokenizer, args.max_length)
    training_args = TrainingArguments(
        output_dir=str(out / "trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=True,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=SQLCollator(tokenizer.pad_token_id),
    )
    result = trainer.train()
    adapter_dir = out / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    manifest["status"] = "trained"
    manifest["train_metrics"] = result.metrics
    manifest["adapter_dir"] = str(adapter_dir)
    if torch.cuda.is_available():
        manifest["max_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated())
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"adapter: {adapter_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
