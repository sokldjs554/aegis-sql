#!/usr/bin/env python3
"""Evaluate a pretrained/LoRA HuggingFace model on KorFin-Bench answerable SQL.

The evaluator intentionally reuses AEGIS normalisation, schema linking, schema
cards, policy guard, SQL executor and execution-match metric.  This keeps the
model comparison focused on generation quality rather than changing retrieval
or scoring at the same time.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

from aegis_sql.config import PROJECT_ROOT, get_settings
from aegis_sql.eval.harness import load_benchmark
from aegis_sql.eval.metrics import execution_match
from aegis_sql.pipeline import AegisEngine
from aegis_sql.training.hf_experiment import SYSTEM_PROMPT, PreparedExample, user_prompt

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
_CODE_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _clean_sql(text: str) -> str:
    match = _CODE_FENCE.search(text)
    if match:
        text = match.group(1)
    text = text.strip()
    starts = [text.upper().find(token) for token in ("SELECT", "WITH")]
    starts = [i for i in starts if i >= 0]
    if starts:
        text = text[min(starts) :]
    if ";" in text:
        text = text.split(";", 1)[0] + ";"
    return text.strip()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate Qwen/HF Text-to-SQL with AEGIS retrieval + EX")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default="", help="PEFT adapter directory; empty evaluates the base model")
    ap.add_argument("--benchmark", default="data/benchmark/korfin_bench.jsonl")
    ap.add_argument("--out", default="reports/hf-qwen-eval.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--load-4bit", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise SystemExit('HuggingFace experiment dependencies missing; run `pip install -e ".[train,hf]"`') from exc

    if args.load_4bit and not torch.cuda.is_available():
        raise SystemExit("--load-4bit requires CUDA")

    quantization = None
    if args.load_4bit:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        quantization_config=quantization,
        device_map="auto" if args.load_4bit else None,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, _resolve(args.adapter))
    elif torch.cuda.is_available() and not args.load_4bit:
        model = model.cuda()
    model.eval()

    settings = get_settings()
    engine = AegisEngine.build(settings)
    items = [i for i in load_benchmark(_resolve(args.benchmark)) if i.expect == "ok"]
    if args.limit is not None:
        items = items[: args.limit]

    rows: list[dict] = []
    correct = Counter()
    total = Counter()
    latencies: list[float] = []

    try:
        for n, item in enumerate(items, 1):
            nq = engine.c.normalizer.normalize(item.question)
            linked = engine.c.linker.link(nq)
            card = engine.c.card_builder.render(linked, style="slm", include_code_dict=False)
            example = PreparedExample(item.question, item.gold_sql or "", card, item.difficulty, "", "benchmark")
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(example)},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            started = time.perf_counter()
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            latency_ms = (time.perf_counter() - started) * 1000
            latencies.append(latency_ms)
            new_tokens = generated[0, inputs["input_ids"].shape[1] :]
            pred_sql = _clean_sql(tokenizer.decode(new_tokens, skip_special_tokens=True))

            ok = False
            error = ""
            verdict = engine.c.guard.check(pred_sql) if pred_sql else None
            if not pred_sql:
                error = "empty generation"
            elif verdict is None or not verdict.allowed:
                error = "policy blocked generated SQL"
            else:
                result = engine.c.executor.execute(verdict.rewritten_sql or pred_sql)
                if not result.ok:
                    error = result.error or "execution failed"
                else:
                    gold = engine.c.executor.execute(item.gold_sql or "")
                    ok = execution_match(result, gold, item.gold_sql or "")
                    if not ok:
                        error = "result mismatch"

            total["all"] += 1
            total[item.difficulty] += 1
            if ok:
                correct["all"] += 1
                correct[item.difficulty] += 1
            rows.append(
                {
                    "id": item.id,
                    "difficulty": item.difficulty,
                    "correct": ok,
                    "pred_sql": pred_sql,
                    "gold_sql": item.gold_sql,
                    "latency_ms": round(latency_ms, 2),
                    "linked_tables": linked.tables,
                    "error": error,
                }
            )
            print(f"[{n:>3}/{len(items)}] {item.id} {'OK' if ok else 'MISS'}")
    finally:
        engine.close()

    def ratio(key: str) -> float:
        return correct[key] / total[key] if total[key] else 0.0

    ordered_latency = sorted(latencies)
    p50 = ordered_latency[len(ordered_latency) // 2] if ordered_latency else 0.0
    p95 = ordered_latency[min(len(ordered_latency) - 1, int(len(ordered_latency) * 0.95))] if ordered_latency else 0.0
    report = {
        "model": args.model,
        "adapter": args.adapter or None,
        "items": len(rows),
        "execution_accuracy": ratio("all"),
        "by_difficulty": {
            k: {"correct": correct[k], "total": total[k], "ex": ratio(k)}
            for k in ("easy", "medium", "hard")
        },
        "latency_ms": {"p50": round(p50, 2), "p95": round(p95, 2)},
        "rows": rows,
    }
    if torch.cuda.is_available():
        report["max_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated())

    out = _resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"EX: {correct['all']}/{total['all']} = {ratio('all'):.1%}")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
