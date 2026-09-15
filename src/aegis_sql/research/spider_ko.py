"""Utilities for an external Korean Spider evaluation.

KorFin-Bench is intentionally domain-specific and lives inside this repository.
Spider-KO gives the project a second, externally curated axis: Korean questions
against database schemas that were never part of the insurance demo.  This
module stays independent from the production AEGIS engine so external benchmark
results cannot silently inherit insurance-specific dictionaries or policies.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aegis_sql.eval.metrics import execution_match
from aegis_sql.schema.card import SchemaCardBuilder, Style
from aegis_sql.schema.introspect import SQLiteIntrospector
from aegis_sql.verify.executor import SQLExecutor

SPIDER_KO_DATASET = "huggingface-KREW/spider-ko"
SPIDER_KO_SPLIT = "validation"
SPIDER_KO_DEV_ITEMS = 1034
_DB_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_REPAIRABLE_SQLITE_ERRORS: tuple[tuple[str, str], ...] = (
    ("no_such_column", "no such column"),
    ("no_such_table", "no such table"),
    ("ambiguous_column", "ambiguous column name"),
    ("syntax_error", "syntax error"),
)


@dataclass(frozen=True, slots=True)
class SpiderKoExample:
    """One Spider-KO example using Korean text as the model input."""

    db_id: str
    question: str
    gold_sql: str
    question_en: str = ""


@dataclass(frozen=True, slots=True)
class ExecutionComparison:
    """Result of executing prediction and gold SQL on the same external DB."""

    ok: bool
    match: bool
    pred_error: str = ""
    gold_error: str = ""


def _records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            return [dict(row) for row in csv.DictReader(fh)]
    if suffix in {".jsonl", ".ndjson"}:
        out: list[dict[str, Any]] = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"row {line_no}: expected a JSON object")
            out.append(payload)
        return out
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
            raise ValueError("JSON dataset must be a list of objects")
        return list(payload)
    raise ValueError(f"unsupported Spider-KO dataset format: {path.suffix}")


def load_spider_ko(path: str | Path, *, limit: int | None = None) -> list[SpiderKoExample]:
    """Load a local Spider-KO export and require the Korean question field.

    The evaluator never falls back to the English ``question`` column.  A
    missing translation is a benchmark-data error, not permission to change
    the language being measured.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Spider-KO dataset not found: {source}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    examples: list[SpiderKoExample] = []
    for index, row in enumerate(_records(source), 1):
        db_id = str(row.get("db_id") or "").strip()
        question_ko = str(row.get("question_ko") or "").strip()
        gold_sql = str(row.get("query") or "").strip()
        if not db_id:
            raise ValueError(f"row {index}: db_id is required")
        if not question_ko:
            raise ValueError(f"row {index}: question_ko is required")
        if not gold_sql:
            raise ValueError(f"row {index}: query is required")
        examples.append(
            SpiderKoExample(
                db_id=db_id,
                question=question_ko,
                gold_sql=gold_sql,
                question_en=str(row.get("question") or "").strip(),
            )
        )
        if limit is not None and len(examples) >= limit:
            break
    return examples


def resolve_spider_db(db_root: str | Path, db_id: str) -> Path:
    """Resolve the official ``database/<db_id>/<db_id>.sqlite`` layout safely."""
    if not _DB_ID.fullmatch(db_id) or ".." in db_id:
        raise ValueError(f"invalid Spider db_id: {db_id!r}")

    root = Path(db_root).resolve()
    candidates = (
        root / db_id / f"{db_id}.sqlite",
        root / "database" / db_id / f"{db_id}.sqlite",
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:  # pragma: no cover - regex already blocks traversal
            raise ValueError(f"db_id escapes database root: {db_id!r}") from exc
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        f"Spider database for {db_id!r} not found under {root}; expected <root>/{db_id}/{db_id}.sqlite"
    )


def spider_schema_card(db_path: str | Path, *, style: Style = "slm") -> str:
    """Render a schema card from an unseen Spider SQLite database."""
    graph = SQLiteIntrospector(db_path).introspect(with_row_counts=False)
    return SchemaCardBuilder(graph).render(style=style, include_code_dict=False)


def bounded_repair_reason(error: str, *, attempts: int = 0) -> str | None:
    """Classify a narrow SQLite execution failure eligible for one repair call."""
    if attempts >= 1:
        return None
    normalized = str(error or "").strip().lower()
    if not normalized:
        return None
    for reason, marker in _REPAIRABLE_SQLITE_ERRORS:
        if marker in normalized:
            return reason
    return None


def validate_bounded_repair_experiment(*, enabled: bool, schema_style: Style) -> None:
    """Keep the repair experiment single-variable by pinning the winning schema style."""
    if enabled and schema_style != "mschema":
        raise ValueError("bounded repair requires schema_style='mschema'")


def build_bounded_repair_prompt(
    *,
    question: str,
    schema_card: str,
    initial_sql: str,
    execution_error: str,
) -> str:
    """Build a one-shot repair prompt from runtime evidence only.

    Gold SQL is intentionally absent from both the function signature and the
    prompt.  The model may use only the original Korean question, the same
    schema representation, its failed SQL, and SQLite's execution error.
    """
    return (
        "첫 SQL이 SQLite 실행에 실패했습니다. 아래 실행 오류만 근거로 SQL을 한 번 수정하세요.\n"
        "스키마에 실제로 존재하는 테이블과 컬럼만 사용하고, 질문의 의미는 바꾸지 마세요.\n"
        "설명이나 마크다운 없이 단일 읽기 전용 SQLite SQL만 출력하세요.\n\n"
        f"### 스키마\n{schema_card}\n"
        f"### 질문\n{question}\n"
        f"### 실패한 SQL\n{initial_sql}\n"
        f"### SQLite 실행 오류\n{execution_error}\n"
        "### 수정 SQL"
    )


def compare_execution(
    db_path: str | Path,
    pred_sql: str,
    gold_sql: str,
    *,
    timeout_s: float = 10.0,
    max_rows: int = 100_000,
) -> ExecutionComparison:
    """Execute both SQL statements read-only and compare their result sets."""
    with SQLExecutor(db_path, timeout_s=timeout_s, max_rows=max_rows) as executor:
        pred = executor.execute(pred_sql)
        gold = executor.execute(gold_sql)
    if not pred.ok or not gold.ok:
        return ExecutionComparison(
            ok=False,
            match=False,
            pred_error=pred.error or "" if pred else "prediction did not execute",
            gold_error=gold.error or "" if gold else "gold did not execute",
        )
    return ExecutionComparison(ok=True, match=execution_match(pred, gold, gold_sql))


def portfolio_evidence_ready(
    report: dict[str, Any], *, expected_items: int = SPIDER_KO_DEV_ITEMS
) -> bool:
    """Return True only for a complete, auditable Korean external-dev run."""
    if not report.get("external_benchmark"):
        return False
    if report.get("benchmark") != SPIDER_KO_DATASET:
        return False
    if report.get("split") != SPIDER_KO_SPLIT or report.get("question_language") != "ko":
        return False
    if report.get("items") != expected_items:
        return False
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != expected_items:
        return False
    required = {"db_id", "question", "gold_sql", "pred_sql", "correct"}
    return all(isinstance(row, dict) and required.issubset(row) for row in rows)


__all__ = [
    "ExecutionComparison",
    "SPIDER_KO_DATASET",
    "SPIDER_KO_DEV_ITEMS",
    "SPIDER_KO_SPLIT",
    "SpiderKoExample",
    "bounded_repair_reason",
    "build_bounded_repair_prompt",
    "compare_execution",
    "load_spider_ko",
    "portfolio_evidence_ready",
    "resolve_spider_db",
    "spider_schema_card",
    "validate_bounded_repair_experiment",
]
