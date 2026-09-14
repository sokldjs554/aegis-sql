"""Regression checks for the deterministic public-demo dataset."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_builder():
    script = ROOT / "scripts" / "build_demo_db.py"
    spec = importlib.util.spec_from_file_location("aegis_build_demo_db", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_deployed_scale_has_distinct_marketing_counts_for_every_branch():
    """The 0.5x Render dataset must make all 24 session scopes observable."""
    module = _load_builder()

    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript((ROOT / "data" / "demo" / "schema.sql").read_text(encoding="utf-8"))
        module.build(conn, scale=0.5)
        rows = conn.execute(
            """
            SELECT a.BRCH_CD, COUNT(*) AS cnt
            FROM TB_CTRT t
            JOIN TB_AGNT a ON a.AGNT_ID = t.AGNT_ID
            JOIN TB_CUST c ON c.CUST_ID = t.CUST_ID
            WHERE c.MKT_AGR_YN = 'Y'
            GROUP BY a.BRCH_CD
            ORDER BY a.BRCH_CD
            """
        ).fetchall()
    finally:
        conn.close()

    counts = [count for _branch, count in rows]
    assert len(rows) == 24
    assert all(count > 0 for count in counts)
    assert len(set(counts)) == 24


def test_force_rebuild_removes_sqlite_sidecars(tmp_path):
    module = _load_builder()
    database = tmp_path / "demo.sqlite"
    artifacts = [
        database,
        Path(f"{database}-journal"),
        Path(f"{database}-wal"),
        Path(f"{database}-shm"),
    ]
    for artifact in artifacts:
        artifact.write_bytes(b"stale")

    module.remove_sqlite_artifacts(database)

    assert not any(artifact.exists() for artifact in artifacts)


def test_full_scale_generator_matches_every_frozen_gold_result():
    module = _load_builder()
    benchmark_path = ROOT / "data" / "benchmark" / "korfin_bench.jsonl"
    items = [
        json.loads(line)
        for line in benchmark_path.read_text(encoding="utf-8").splitlines()
        if line
    ]

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(
            (ROOT / "data" / "demo" / "schema.sql").read_text(encoding="utf-8")
        )
        module.build(conn, scale=1.0)
        for item in items:
            if item["expect"] != "ok":
                continue
            cursor = conn.execute(item["gold_sql"])
            rows = cursor.fetchall()
            columns = [column[0] for column in cursor.description or ()]
            preview = [list(row) for row in rows[:3]]
            assert len(rows) == item["gold_row_count"], item["id"]
            assert columns == item["gold_columns"], item["id"]
            assert preview == item["gold_preview"], item["id"]
    finally:
        conn.close()
