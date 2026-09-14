"""Contract tests for the external Spider-KO evaluation path."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest


def _write_fixture_dataset(path: Path) -> None:
    rows = [
        {
            "db_id": "school",
            "query": "SELECT COUNT(*) FROM students",
            "question": "How many students are there?",
            "question_ko": "학생은 몇 명인가요?",
        },
        {
            "db_id": "school",
            "query": "SELECT name FROM students ORDER BY name",
            "question": "List student names.",
            "question_ko": "학생 이름을 정렬해서 보여줘",
        },
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _write_fixture_db(root: Path) -> Path:
    db_dir = root / "school"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "school.sqlite"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE departments (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            );
            CREATE TABLE students (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                department_id INTEGER,
                FOREIGN KEY (department_id) REFERENCES departments(id)
            );
            INSERT INTO departments VALUES (1, 'AI');
            INSERT INTO students VALUES (1, 'Kim', 1), (2, 'Lee', 1);
            """
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


def test_loader_uses_korean_question_and_keeps_gold_sql(tmp_path: Path):
    from aegis_sql.research.spider_ko import load_spider_ko

    dataset = tmp_path / "validation.jsonl"
    _write_fixture_dataset(dataset)

    rows = load_spider_ko(dataset)

    assert len(rows) == 2
    assert rows[0].question == "학생은 몇 명인가요?"
    assert rows[0].question_en == "How many students are there?"
    assert rows[0].gold_sql == "SELECT COUNT(*) FROM students"
    assert rows[0].db_id == "school"


def test_loader_rejects_rows_without_korean_question(tmp_path: Path):
    from aegis_sql.research.spider_ko import load_spider_ko

    dataset = tmp_path / "validation.jsonl"
    dataset.write_text(
        json.dumps({"db_id": "school", "query": "SELECT 1", "question": "English only"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="question_ko"):
        load_spider_ko(dataset)


def test_resolve_db_supports_official_spider_layout_and_blocks_traversal(tmp_path: Path):
    from aegis_sql.research.spider_ko import resolve_spider_db

    db_root = tmp_path / "database"
    db_path = _write_fixture_db(db_root)

    assert resolve_spider_db(db_root, "school") == db_path
    with pytest.raises(ValueError, match="db_id"):
        resolve_spider_db(db_root, "../escape")


def test_external_schema_card_is_built_from_unseen_sqlite_schema(tmp_path: Path):
    from aegis_sql.research.spider_ko import spider_schema_card

    db_path = _write_fixture_db(tmp_path / "database")
    card = spider_schema_card(db_path, style="slm")

    assert "students(" in card
    assert "departments(" in card
    assert "students.department_id=departments.id" in card


def test_execution_match_uses_each_external_database(tmp_path: Path):
    from aegis_sql.research.spider_ko import compare_execution

    db_path = _write_fixture_db(tmp_path / "database")
    same = compare_execution(db_path, "SELECT COUNT(*) FROM students", "SELECT COUNT(id) FROM students")
    different = compare_execution(db_path, "SELECT COUNT(*) FROM students", "SELECT COUNT(*) FROM departments")

    assert same.ok is True
    assert same.match is True
    assert different.ok is True
    assert different.match is False


def test_portfolio_ready_requires_full_external_dev_and_row_level_audit():
    from aegis_sql.research.spider_ko import SPIDER_KO_DEV_ITEMS, portfolio_evidence_ready

    base = {
        "external_benchmark": True,
        "benchmark": "huggingface-KREW/spider-ko",
        "split": "validation",
        "question_language": "ko",
        "items": SPIDER_KO_DEV_ITEMS,
        "rows": [
            {"db_id": "db", "question": "질문", "gold_sql": "SELECT 1", "pred_sql": "SELECT 1", "correct": True}
            for _ in range(SPIDER_KO_DEV_ITEMS)
        ],
    }

    assert portfolio_evidence_ready(base) is True
    assert portfolio_evidence_ready(base | {"items": 10, "rows": base["rows"][:10]}) is False
    assert portfolio_evidence_ready(base | {"question_language": "en"}) is False
    incomplete = dict(base)
    incomplete["rows"] = [{"db_id": "db", "question": "질문", "gold_sql": "SELECT 1", "correct": True}] * SPIDER_KO_DEV_ITEMS
    assert portfolio_evidence_ready(incomplete) is False
