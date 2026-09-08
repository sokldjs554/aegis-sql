"""Shared preparation utilities for the LitE-SQL-inspired HuggingFace baseline.

The large checkpoint itself is intentionally an optional research dependency.
Everything in this module runs with the normal AEGIS install, so CI can verify
that the Qwen experiment uses the *same* flywheel records and schema-card logic
as the in-house sLLM without downloading a multi-gigabyte model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aegis_sql.schema.card import SchemaCardBuilder
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
    style: str = "slm",
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
