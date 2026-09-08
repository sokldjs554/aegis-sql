from __future__ import annotations

import json
from pathlib import Path

from aegis_sql.nlu.answerability import SchemaAnswerabilityDetector

ROOT = Path(__file__).resolve().parents[1]
PROBES = ROOT / "data" / "research" / "ehrsql_unanswerable_probes.jsonl"


def _rows() -> list[dict]:
    return [json.loads(line) for line in PROBES.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_ehrsql_inspired_probe_set_matches_actual_schema(schema, glossary):
    """The curated research set must be evaluated against the real demo schema.

    This is intentionally not a generalisation claim: the 30 questions were
    constructed to test one missing evaluation axis (clear but unavailable
    information).  The test proves that the capability registry is anchored to
    actual schema/glossary evidence and does not reject the paired answerable
    hard negatives.
    """

    detector = SchemaAnswerabilityDetector(schema, glossary.entries)
    rows = _rows()
    assert len(rows) == 30

    tp = tn = fp = fn = 0
    for row in rows:
        decision = detector.assess(row["question"])
        predicted_unanswerable = not decision.answerable
        expected_unanswerable = row["expect"] == "unanswerable"

        if expected_unanswerable:
            assert row["missing_concept"] in decision.missing_concepts, row["id"]

        if expected_unanswerable and predicted_unanswerable:
            tp += 1
        elif expected_unanswerable:
            fn += 1
        elif predicted_unanswerable:
            fp += 1
        else:
            tn += 1

    assert (tp, tn, fp, fn) == (15, 15, 0, 0)


def test_schema_evidence_suppresses_a_capability_gap(schema, glossary):
    """A rule must stop refusing a concept once the schema exposes evidence for it."""

    from aegis_sql.nlu.answerability import CapabilityRule
    from aegis_sql.types import ColumnInfo, TableInfo

    table = TableInfo(name="TB_EXT", comment="외부 신용정보")
    table.columns.append(ColumnInfo(table="TB_EXT", name="CREDIT_SCORE", dtype="INTEGER", comment="신용점수"))
    schema.tables[table.name] = table
    try:
        detector = SchemaAnswerabilityDetector(
            schema,
            glossary.entries,
            rules=(
                CapabilityRule(
                    "customer_credit_score",
                    (("신용점수",),),
                    (("신용점수",),),
                    "신용점수 없음",
                ),
            ),
        )
        assert detector.assess("신용점수가 낮은 고객 수").answerable
    finally:
        schema.tables.pop(table.name, None)
