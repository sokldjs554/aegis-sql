#!/usr/bin/env python3
"""Evaluate a HuggingFace causal LM on the external Spider-KO dev split.

Unlike KorFin-Bench, every item here targets a Spider database outside the
insurance demo.  The evaluator uses the Korean ``question_ko`` text, introspects
the corresponding SQLite schema, generates one SQL statement, and compares the
prediction with the gold query by executing both against that same database.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from aegis_sql.eval.metrics import execution_match
from aegis_sql.research.spider_ko import (
    SPIDER_KO_DATASET,
    SPIDER_KO_SPLIT,
    load_spider_ko,
    portfolio_evidence_ready,
    resolve_spider_db,
    spider_schema_card,
)
from aegis_sql.training.hf_experiment import file_sha256, git_sha, runtime_metadata
from aegis_sql.verify.executor import SQLExecutor

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
SYSTEM_PROMPT = (
    "당신은 관계형 데이터베이스용 Text-to-SQL 모델입니다. "
    "제공된 스키마와 질문만 사용해 질문에 답하는 단일 읽기 전용 SQLite SQL을 생성하세요. "
    "설명과 마크다운은 출력하지 마세요."
)
_CODE_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def clean_sql(text: str) -> str:
    match = _CODE_FENCE.search(text)
    if match:
        text = match.group(1)
    text = text.strip()
    starts = [text.upper().find(token) for token in ("SELECT", "WITH")]
    starts = [index for index in starts if index >= 0]
    if starts:
        text = text[min(starts) :]
    if ";" in text:
        text = text.split(";", 1)[0] + ";"
    return text.strip()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * q))
    return ordered[index]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="External Korean Text-to-SQL evaluation on Spider-KO")
    ap.add_argument("--dataset", required=True, help="Local Spider-KO validation CSV/JSONL export")
    ap.add_argument("--db-root", required=True, help="Root containing <db_id>/<db_id>.sqlite Spider DBs")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default="", help="Optional PEFT adapter directory")
    ap.add_argument("--out", default="reports/spider-ko-hf-eval.json")
    ap.add_argument("--limit", type=int, default=None, help="Smoke only; full evidence requires all 1,034 dev rows")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--warmup-runs", type=int, default=1)
    ap.add_argument("--schema-style", choices=("slm", "ddl", "compact", "mschema"), default="slm")
    ap.add_argument("--load-4bit", action="store_true")
    return ap.parse_args()


def main() -> int:  # noqa: C901 - one explicit research loop keeps the evidence path inspectable
    args = parse_args()
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise SystemExit('HF dependencies missing; run `pip install -e ".[hf]"`') from exc

    if args.load_4bit and not torch.cuda.is_available():
        raise SystemExit("--load-4bit requires CUDA")
    if args.warmup_runs < 0:
        raise SystemExit("--warmup-runs must be non-negative")

    dataset_path = Path(args.dataset).resolve()
    db_root = Path(args.db_root).resolve()
    examples = load_spider_ko(dataset_path, limit=args.limit)
    if not examples:
        raise SystemExit("Spider-KO dataset contains no rows")

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
        model = PeftModel.from_pretrained(model, args.adapter)
    elif torch.cuda.is_available() and not args.load_4bit:
        model = model.cuda()
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    schema_cards: dict[str, str] = {}
    db_paths: dict[str, Path] = {}
    db_hashes: dict[str, str] = {}
    executors: dict[str, SQLExecutor] = {}
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    correct = 0
    execution_failures = 0

    try:
        for n, item in enumerate(examples, 1):
            if item.db_id not in db_paths:
                path = resolve_spider_db(db_root, item.db_id)
                db_paths[item.db_id] = path
                db_hashes[item.db_id] = file_sha256(path)
                schema_cards[item.db_id] = spider_schema_card(path, style=args.schema_style)
                executors[item.db_id] = SQLExecutor(path, timeout_s=10.0, max_rows=100_000)

            prompt = (
                f"### 스키마\n{schema_cards[item.db_id]}\n"
                f"### 질문\n{item.question}\n### SQL"
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
            device = next(model.parameters()).device
            inputs = {key: value.to(device) for key, value in inputs.items()}
            generation_args = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": False,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }

            if n == 1 and args.warmup_runs:
                with torch.inference_mode():
                    for _ in range(args.warmup_runs):
                        model.generate(**inputs, **generation_args)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**inputs, **generation_args)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - started) * 1000.0
            latencies.append(latency_ms)
            new_tokens = generated[0, inputs["input_ids"].shape[1] :]
            pred_sql = clean_sql(tokenizer.decode(new_tokens, skip_special_tokens=True))

            executor = executors[item.db_id]
            pred_result = executor.execute(pred_sql) if pred_sql else None
            gold_result = executor.execute(item.gold_sql)
            item_correct = execution_match(pred_result, gold_result, item.gold_sql)
            if item_correct:
                correct += 1
            if pred_result is None or not pred_result.ok or not gold_result.ok:
                execution_failures += 1

            rows.append(
                {
                    "db_id": item.db_id,
                    "question": item.question,
                    "question_en": item.question_en,
                    "gold_sql": item.gold_sql,
                    "pred_sql": pred_sql,
                    "correct": item_correct,
                    "latency_ms": round(latency_ms, 2),
                    "pred_execution_ok": bool(pred_result and pred_result.ok),
                    "pred_error": "" if pred_result is None else (pred_result.error or ""),
                    "gold_execution_ok": gold_result.ok,
                    "gold_error": gold_result.error or "",
                }
            )
            print(f"[{n:>4}/{len(examples)}] {item.db_id:<24} {'OK' if item_correct else 'MISS'}")
    finally:
        for executor in executors.values():
            executor.close()

    report: dict[str, Any] = {
        "external_benchmark": True,
        "benchmark": SPIDER_KO_DATASET,
        "split": SPIDER_KO_SPLIT,
        "question_language": "ko",
        "model": args.model,
        "adapter": args.adapter or None,
        "items": len(rows),
        "correct": correct,
        "execution_accuracy": correct / len(rows),
        "execution_failures": execution_failures,
        "latency_ms": {
            "p50": round(percentile(latencies, 0.50), 2),
            "p95": round(percentile(latencies, 0.95), 2),
        },
        "evaluation": {
            "git_sha": git_sha(),
            "dataset_path": str(dataset_path),
            "dataset_sha256": file_sha256(dataset_path),
            "db_root": str(db_root),
            "database_sha256": db_hashes,
            "schema_style": args.schema_style,
            "limit": args.limit,
            "max_new_tokens": args.max_new_tokens,
            "warmup_runs": args.warmup_runs,
            "quantization": "NF4 4-bit" if args.load_4bit else "none",
            "latency_scope": "model.generate only; CUDA synchronized; warmup excluded",
            "comparison": "AEGIS execution_match on the official per-db SQLite database",
        },
        "runtime": runtime_metadata(torch),
        "rows": rows,
    }
    if torch.cuda.is_available():
        report["max_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated())
        report["max_cuda_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    report["portfolio_evidence_ready"] = portfolio_evidence_ready(report)

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"EX: {correct}/{len(rows)} = {report['execution_accuracy']:.1%}")
    print(f"portfolio_evidence_ready: {report['portfolio_evidence_ready']}")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
