"""Regression checks for the deterministic public-demo dataset."""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_deployed_scale_has_distinct_marketing_counts_for_every_branch():
    """The 0.5x Render dataset must make all 24 session scopes observable."""
    script = ROOT / "scripts" / "build_demo_db.py"
    spec = importlib.util.spec_from_file_location("aegis_build_demo_db", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

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
