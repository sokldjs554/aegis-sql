#!/usr/bin/env python3
"""Evaluate a HuggingFace causal LM on the external Spider-KO dev split.

Unlike KorFin-Bench, every item here targets a Spider database outside the
insurance demo.  The evaluator uses the Korean ``question_ko`` text, introspects
the corresponding SQLite schema, generates one SQL statement, and compares the
prediction with the gold query by executing both against that same database.

When ``--bounded-repair`` is enabled, the experiment keeps the winning
``mschema`` representation fixed and permits at most one additional generation
for a narrow set of SQLite execution errors.  The repair prompt never receives
the gold SQL, and the repair decision is made before the gold query is executed.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from aegis_sql.eval.metrics import execution_match
from aegis_sql.research.spider_ko import (
    SPIDER_KO_DATASET,
    SPIDER_KO_SPLIT,
    bounded_repair_reason,
    build_bounded_repair_prompt,
    load_spider_ko,
    portfolio_evidence_ready,
    resolve_spider_db,
    spider_schema_card,
    validate_bounded_repair_experiment,
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
    ap.add_argument(
        "--bounded-repair",
        action="store_true",
        help="For mschema only, retry one repairable SQLite execution failure exactly once",
    )
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
    try:
        validate_bounded_repair_experiment(
            enabled=args.bounded_repair,
            schema_style=args.schema_style,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

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

    generation_args = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    device = next(model.parameters()).device

    def generate_sql(user_prompt: str, *, warmup_runs: int = 0) -> tuple[str, float]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        if warmup_runs:
            with torch.inference_mode():
                for _ in range(warmup_runs):
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
        new_tokens = generated[0, inputs["input_ids"].shape[1] :]
        sql = clean_sql(tokenizer.decode(new_tokens, skip_special_tokens=True))
        return sql, latency_ms

    schema_cards: dict[str, str] = {}
    db_paths: dict[str, Path] = {}
    db_hashes: dict[str, str] = {}
    executors: dict[str, SQLExecutor] = {}
    rows: list[dict[str, Any]] = []
    initial_latencies: list[float] = []
    total_generation_latencies: list[float] = []
    repair_latencies: list[float] = []
    repair_reasons: Counter[str] = Counter()

    initial_correct = 0
    initial_execution_failures = 0
    correct = 0
    execution_failures = 0
    repair_attempted = 0
    repair_execution_recovered = 0
    repair_correct = 0

    try:
        for n, item in enumerate(examples, 1):
            if item.db_id not in db_paths:
                path = resolve_spider_db(db_root, item.db_id)
                db_paths[item.db_id] = path
                db_hashes[item.db_id] = file_sha256(path)
                schema_cards[item.db_id] = spider_schema_card(path, style=args.schema_style)
                executors[item.db_id] = SQLExecutor(path, timeout_s=10.0, max_rows=100_000)

            schema_card = schema_cards[item.db_id]
            prompt = f"### 스키마\n{schema_card}\n### 질문\n{item.question}\n### SQL"
            initial_pred_sql, initial_latency_ms = generate_sql(
                prompt,
                warmup_runs=args.warmup_runs if n == 1 else 0,
            )
            initial_latencies.append(initial_latency_ms)

            executor = executors[item.db_id]
            initial_pred_result = executor.execute(initial_pred_sql) if initial_pred_sql else None
            pred_sql = initial_pred_sql
            pred_result = initial_pred_result
            item_repair_attempted = False
            item_repair_reason = ""
            repaired_sql: str | None = None
            repair_latency_ms = 0.0
            item_repair_execution_recovered = False

            initial_error = "" if initial_pred_result is None else (initial_pred_result.error or "")
            reason = (
                bounded_repair_reason(initial_error, attempts=0)
                if (
                    args.bounded_repair
                    and initial_pred_result is not None
                    and not initial_pred_result.ok
                )
                else None
            )
            if reason is not None:
                item_repair_attempted = True
                item_repair_reason = reason
                repair_attempted += 1
                repair_reasons[reason] += 1

                repair_prompt = build_bounded_repair_prompt(
                    question=item.question,
                    schema_card=schema_card,
                    initial_sql=initial_pred_sql,
                    execution_error=initial_error,
                )
                repaired_sql, repair_latency_ms = generate_sql(repair_prompt)
                repair_latencies.append(repair_latency_ms)
                pred_sql = repaired_sql
                pred_result = executor.execute(repaired_sql) if repaired_sql else None
                item_repair_execution_recovered = bool(pred_result and pred_result.ok)
                if item_repair_execution_recovered:
                    repair_execution_recovered += 1

            # Gold is executed only after the generation path is final.  It is
            # evaluation evidence, never an input or gate for the repair policy.
            gold_result = executor.execute(item.gold_sql)
            initial_item_correct = execution_match(
                initial_pred_result,
                gold_result,
                item.gold_sql,
            )
            item_correct = execution_match(pred_result, gold_result, item.gold_sql)
            item_repair_correct = bool(item_repair_attempted and item_correct)

            if initial_item_correct:
                initial_correct += 1
            if initial_pred_result is None or not initial_pred_result.ok or not gold_result.ok:
                initial_execution_failures += 1
            if item_repair_correct:
                repair_correct += 1

            total_generation_latency_ms = initial_latency_ms + repair_latency_ms
            total_generation_latencies.append(total_generation_latency_ms)
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
                    "initial_pred_sql": initial_pred_sql,
                    "initial_correct": initial_item_correct,
                    "initial_pred_execution_ok": bool(initial_pred_result and initial_pred_result.ok),
                    "initial_pred_error": initial_error,
                    "repair_attempted": item_repair_attempted,
                    "repair_reason": item_repair_reason,
                    "repaired_sql": repaired_sql,
                    "repair_latency_ms": round(repair_latency_ms, 2),
                    "repair_execution_recovered": item_repair_execution_recovered,
                    "repair_correct": item_repair_correct,
                    "pred_sql": pred_sql,
                    "correct": item_correct,
                    "latency_ms": round(initial_latency_ms, 2),
                    "total_generation_latency_ms": round(total_generation_latency_ms, 2),
                    "pred_execution_ok": bool(pred_result and pred_result.ok),
                    "pred_error": "" if pred_result is None else (pred_result.error or ""),
                    "gold_execution_ok": gold_result.ok,
                    "gold_error": gold_result.error or "",
                }
            )
            suffix = f" repair={item_repair_reason}" if item_repair_attempted else ""
            print(
                f"[{n:>4}/{len(examples)}] {item.db_id:<24} "
                f"{'OK' if item_correct else 'MISS'}{suffix}"
            )
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
        "initial_correct": initial_correct,
        "initial_execution_accuracy": initial_correct / len(rows),
        "initial_execution_failures": initial_execution_failures,
        "correct": correct,
        "execution_accuracy": correct / len(rows),
        "execution_failures": execution_failures,
        "latency_ms": {
            "p50": round(percentile(initial_latencies, 0.50), 2),
            "p95": round(percentile(initial_latencies, 0.95), 2),
        },
        "total_generation_latency_ms": {
            "p50": round(percentile(total_generation_latencies, 0.50), 2),
            "p95": round(percentile(total_generation_latencies, 0.95), 2),
        },
        "repair": {
            "enabled": args.bounded_repair,
            "max_attempts": 1,
            "attempted": repair_attempted,
            "execution_recovered": repair_execution_recovered,
            "correct_after_repair": repair_correct,
            "reasons": dict(sorted(repair_reasons.items())),
            "latency_ms": {
                "p50": round(percentile(repair_latencies, 0.50), 2),
                "p95": round(percentile(repair_latencies, 0.95), 2),
            },
            "repair_decision_uses_gold": False,
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
            "bounded_repair": args.bounded_repair,
            "repair_max_attempts": 1,
            "repairable_errors": [
                "no such column",
                "no such table",
                "ambiguous column name",
                "syntax error",
            ],
            "latency_scope": (
                "latency_ms = initial model.generate only; "
                "total_generation_latency_ms = initial + repair generation; "
                "CUDA synchronized; warmup excluded"
            ),
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
    print(
        f"initial EX: {initial_correct}/{len(rows)} = "
        f"{report['initial_execution_accuracy']:.1%}"
    )
    print(f"final EX: {correct}/{len(rows)} = {report['execution_accuracy']:.1%}")
    if args.bounded_repair:
        print(
            "repair: "
            f"attempted={repair_attempted}, "
            f"execution_recovered={repair_execution_recovered}, "
            f"correct_after_repair={repair_correct}"
        )
    print(f"portfolio_evidence_ready: {report['portfolio_evidence_ready']}")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
