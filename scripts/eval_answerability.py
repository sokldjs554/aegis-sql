#!/usr/bin/env python3
"""Evaluate the EHRSQL-inspired answerability research set.

This is deliberately separate from the main KorFin-Bench score.  It measures
whether a schema-capability gate can abstain when the requested information does
not exist while preserving nearby answerable questions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aegis_sql.nlu.answerability import SchemaAnswerabilityDetector
from aegis_sql.retrieval.glossary import Glossary
from aegis_sql.schema.introspect import introspect

ROOT = Path(__file__).resolve().parents[1]


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(detector: SchemaAnswerabilityDetector, rows: list[dict]) -> dict:
    tp = tn = fp = fn = 0
    details: list[dict] = []
    for row in rows:
        decision = detector.assess(row["question"])
        predicted_unanswerable = not decision.answerable
        expected_unanswerable = row["expect"] == "unanswerable"
        if expected_unanswerable and predicted_unanswerable:
            tp += 1
        elif expected_unanswerable:
            fn += 1
        elif predicted_unanswerable:
            fp += 1
        else:
            tn += 1
        details.append(
            {
                "id": row["id"],
                "expected": row["expect"],
                "predicted": "answerable" if decision.answerable else "unanswerable",
                "missing_concepts": list(decision.missing_concepts),
                "reasons": list(decision.reasons),
            }
        )

    unanswerable_total = tp + fn
    answerable_total = tn + fp
    return {
        "items": len(rows),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "unanswerable_recall": tp / unanswerable_total if unanswerable_total else 0.0,
        "false_abstention_rate": fp / answerable_total if answerable_total else 0.0,
        "accuracy": (tp + tn) / len(rows) if rows else 0.0,
        "details": details,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run EHRSQL-inspired answerability research probe")
    ap.add_argument("--db", default=str(ROOT / "data" / "demo" / "aegis_demo.sqlite"))
    ap.add_argument(
        "--probes",
        default=str(ROOT / "data" / "research" / "ehrsql_unanswerable_probes.jsonl"),
    )
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    db = Path(args.db)
    if not db.exists():
        raise SystemExit(f"demo database missing: {db} (run `make demo-db` first)")
    probes = Path(args.probes)
    schema = introspect(db)
    glossary_path = db.parent / "glossary.yaml"
    glossary = Glossary.load(glossary_path) if glossary_path.exists() else Glossary(entries=[])
    detector = SchemaAnswerabilityDetector(schema, glossary.entries)
    result = evaluate(detector, load_rows(probes))

    print(f"items: {result['items']}")
    print(f"unanswerable recall: {result['unanswerable_recall']:.1%}")
    print(f"false abstention rate: {result['false_abstention_rate']:.1%}")
    print(f"accuracy: {result['accuracy']:.1%}")
    print("confusion:", result["confusion"])

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
