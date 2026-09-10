"""Shared preparation utilities for the LitE-SQL-inspired HuggingFace baseline.

The large checkpoint itself is intentionally an optional research dependency.
Everything in this module runs with the normal AEGIS install, so CI can verify
that the Qwen experiment uses the *same* flywheel records and schema-card logic
as the in-house sLLM without downloading a multi-gigabyte model.
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from aegis_sql.schema.card import SchemaCardBuilder, Style
from aegis_sql.types import LinkedSchema

SYSTEM_PROMPT = (
    "당신은 보험 레거시 데이터베이스용 Text-to-SQL 모델입니다. "
    "질문에 답하는 단일 읽기 전용 SQL만 생성하세요. 설명과 마크다운은 출력하지 마세요."
)


@dataclass(slots=True)
class PreparedExample:
    question: str
    sql: str
    schema_card: str
    difficulty: str
    template_id: str
    source: str


def load_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load one flywheel split while preserving the committed record schema."""
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
        if limit is not None and len(rows) >= limit:
            break
    return rows


def prepare_records(
    rows: list[dict[str, Any]],
    card_builder: SchemaCardBuilder,
    *,
    style: Style = "slm",
) -> list[PreparedExample]:
    """Render the same AEGIS schema representation for every supervision pair.

    ``tables`` comes from the flywheel program that produced the SQL.  We expose
    all columns of those tables rather than using gold SQL columns, avoiding a
    label leak while keeping the context smaller than the full database.
    """
    out: list[PreparedExample] = []
    for row in rows:
        tables = list(row.get("tables") or [])
        linked = LinkedSchema(tables=tables)
        card = card_builder.render(linked, style=style, include_code_dict=False)
        out.append(
            PreparedExample(
                question=str(row["question"]),
                sql=str(row["sql"]).strip(),
                schema_card=card,
                difficulty=str(row.get("difficulty", "medium")),
                template_id=str(row.get("template_id", "")),
                source=str(row.get("source", "")),
            )
        )
    return out


def user_prompt(example: PreparedExample) -> str:
    return f"### 스키마\n{example.schema_card}\n### 질문\n{example.question}\n### SQL"


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def records_sha256(rows: list[dict[str, Any]]) -> str:
    """Hash the exact selected records independently of their source file."""
    h = hashlib.sha256()
    for row in rows:
        payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        h.update(payload.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def select_records(
    rows: list[dict[str, Any]],
    *,
    limit: int | None,
    seed: int,
    shuffle_before_limit: bool,
) -> list[dict[str, Any]]:
    """Select the same shuffled prefix used by the historical AegisLM run."""
    selected = list(rows)
    if shuffle_before_limit:
        random.Random(seed).shuffle(selected)
    return selected[:limit] if limit is not None else selected


def git_sha() -> str:
    """Return the exact source revision used by an experiment."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def runtime_metadata(torch: Any) -> dict[str, Any]:
    """Collect reproducibility metadata without importing the HF stack in CI."""

    packages: dict[str, str] = {}
    for package in ("transformers", "peft", "accelerate", "bitsandbytes"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not-installed"

    cuda_available = bool(torch.cuda.is_available())
    metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_available": cuda_available,
        "cuda_runtime": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "packages": packages,
    }
    if cuda_available:
        device = int(torch.cuda.current_device())
        props = torch.cuda.get_device_properties(device)
        metadata["gpu"] = {
            "index": device,
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": int(props.total_memory),
            "compute_capability": f"{props.major}.{props.minor}",
        }
    return metadata


def _validated_eval_block(report: dict[str, Any], *, expected_items: int, label: str) -> dict[str, Any]:
    if int(report.get("items", -1)) != expected_items:
        raise ValueError(f"{label}: expected {expected_items} evaluated items, got {report.get('items')}")

    difficulty = report.get("by_difficulty")
    if not isinstance(difficulty, dict) or any(k not in difficulty for k in ("easy", "medium", "hard")):
        raise ValueError(f"{label}: easy/medium/hard breakdown is missing")
    difficulty_total = sum(int(difficulty[k].get("total", 0)) for k in ("easy", "medium", "hard"))
    if difficulty_total != expected_items:
        raise ValueError(
            f"{label}: difficulty totals add up to {difficulty_total}, expected {expected_items}"
        )
    difficulty_correct = sum(int(difficulty[k].get("correct", 0)) for k in ("easy", "medium", "hard"))
    measured_accuracy = float(report["execution_accuracy"])
    if abs(measured_accuracy - difficulty_correct / expected_items) > 1e-9:
        raise ValueError(f"{label}: overall EX does not match the difficulty counts")
    for key in ("easy", "medium", "hard"):
        bucket_total = int(difficulty[key].get("total", 0))
        bucket_correct = int(difficulty[key].get("correct", 0))
        bucket_ex = float(difficulty[key].get("ex", 0.0))
        expected_ex = bucket_correct / bucket_total if bucket_total else 0.0
        if abs(bucket_ex - expected_ex) > 1e-9:
            raise ValueError(f"{label}: {key} EX does not match its counts")

    latency = report.get("latency_ms")
    if not isinstance(latency, dict) or any(k not in latency for k in ("p50", "p95")):
        raise ValueError(f"{label}: p50/p95 latency is missing")
    peak_memory = int(report.get("max_cuda_memory_bytes", 0))
    if peak_memory <= 0:
        raise ValueError(f"{label}: measured CUDA peak memory is missing")

    return {
        "execution_accuracy": measured_accuracy,
        "by_difficulty": difficulty,
        "latency_ms": {"p50": float(latency["p50"]), "p95": float(latency["p95"])},
        "peak_cuda_memory_bytes": peak_memory,
        "peak_cuda_memory_gib": round(peak_memory / (1024**3), 3),
    }


def build_qwen_summary(
    manifest: dict[str, Any],
    base: dict[str, Any],
    adapted: dict[str, Any],
    *,
    expected_items: int,
    run_kind: str,
) -> dict[str, Any]:
    """Validate a measured Qwen run and return its compact evidence record.

    Smoke and full outputs intentionally pass through the same checks, but only
    a complete 90-item full run is marked as usable portfolio evidence.
    """

    if run_kind not in {"smoke", "full"}:
        raise ValueError("run_kind must be 'smoke' or 'full'")
    if manifest.get("status") != "trained":
        raise ValueError("training manifest is not in the trained state")
    if manifest.get("model") != base.get("model") or manifest.get("model") != adapted.get("model"):
        raise ValueError("base, adapted and training manifest model IDs differ")
    if base.get("adapter") not in (None, ""):
        raise ValueError("base report unexpectedly contains an adapter")
    if not adapted.get("adapter"):
        raise ValueError("adapted report does not identify a PEFT adapter")

    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("training manifest dataset metadata is missing")
    source_snapshot = dataset.get("source_snapshot")
    if not isinstance(source_snapshot, dict) or source_snapshot.get("name") != "aegis-qwen-flywheel-v1":
        raise ValueError("the frozen Qwen dataset snapshot is missing")
    if run_kind == "full" and (
        int(dataset.get("train_count", 0)) != 9000 or int(dataset.get("dev_count", 0)) != 1153
    ):
        raise ValueError("full run does not use the complete frozen 9,000/1,153 train/dev snapshot")

    training = manifest.get("training")
    if not isinstance(training, dict) or float(training.get("wall_clock_seconds", 0.0)) <= 0.0:
        raise ValueError("measured QLoRA training time is missing")
    training_peak = int(manifest.get("max_cuda_memory_bytes", 0))
    if training_peak <= 0:
        raise ValueError("measured QLoRA CUDA peak memory is missing")

    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict) or not runtime.get("cuda_available") or not runtime.get("gpu"):
        raise ValueError("training GPU metadata is missing")

    manifest_sha = manifest.get("git_sha")
    training_gpu = runtime["gpu"].get("name")
    for label, report in (("base", base), ("adapted", adapted)):
        evaluation = report.get("evaluation")
        if not isinstance(evaluation, dict) or evaluation.get("git_sha") != manifest_sha:
            raise ValueError(f"{label}: source revision differs from the training manifest")
        if evaluation.get("quantization") != "NF4 4-bit":
            raise ValueError(f"{label}: expected the controlled NF4 4-bit comparison")
        eval_runtime = report.get("runtime")
        eval_gpu = eval_runtime.get("gpu", {}).get("name") if isinstance(eval_runtime, dict) else None
        if not eval_gpu or eval_gpu != training_gpu:
            raise ValueError(f"{label}: GPU differs from the training run or is missing")

    base_block = _validated_eval_block(base, expected_items=expected_items, label="base")
    adapted_block = _validated_eval_block(adapted, expected_items=expected_items, label="adapted")
    base_sha = base.get("evaluation", {}).get("benchmark_sha256")
    adapted_sha = adapted.get("evaluation", {}).get("benchmark_sha256")
    if not base_sha or base_sha != adapted_sha:
        raise ValueError("base and adapted benchmark hashes differ or are missing")
    expected_database_sha = manifest.get("database", {}).get("sha256")
    base_database_sha = base.get("evaluation", {}).get("database_sha256")
    adapted_database_sha = adapted.get("evaluation", {}).get("database_sha256")
    if (
        not expected_database_sha
        or base_database_sha != expected_database_sha
        or adapted_database_sha != expected_database_sha
    ):
        raise ValueError("training and evaluation database hashes differ or are missing")

    base_ex = base_block["execution_accuracy"]
    adapted_ex = adapted_block["execution_accuracy"]
    return {
        "status": "measured",
        "run_kind": run_kind,
        "portfolio_evidence_ready": run_kind == "full" and expected_items == 90,
        "model": manifest["model"],
        "git_sha": manifest.get("git_sha", "unknown"),
        "benchmark_sha256": base_sha,
        "database_sha256": expected_database_sha,
        "dataset": dataset,
        "items": expected_items,
        "runtime": runtime,
        "training": {
            **training,
            "peak_cuda_memory_bytes": training_peak,
            "peak_cuda_memory_gib": round(training_peak / (1024**3), 3),
        },
        "base": base_block,
        "adapted": adapted_block,
        "delta_ex_percentage_points": round((adapted_ex - base_ex) * 100.0, 4),
    }


def experiment_manifest(
    *,
    model: str,
    train_path: str | Path,
    dev_path: str | Path,
    train_count: int,
    dev_count: int,
    seed: int,
    schema_fingerprint: str,
    qlora: bool,
    lora_r: int,
    lora_alpha: int,
) -> dict[str, Any]:
    """The minimum metadata required to tell two fine-tuning runs apart."""
    return {
        "model": model,
        "dataset": {
            "train": str(train_path),
            "train_sha256": file_sha256(train_path),
            "dev": str(dev_path),
            "dev_sha256": file_sha256(dev_path),
            "train_count": train_count,
            "dev_count": dev_count,
        },
        "schema_fingerprint": schema_fingerprint,
        "seed": seed,
        "adapter": {
            "method": "QLoRA" if qlora else "LoRA",
            "r": lora_r,
            "alpha": lora_alpha,
            "targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
    }
