from __future__ import annotations

import json
from pathlib import Path

EVIDENCE = Path("data/research/spider_ko_bounded_repair_evidence.json")


def test_spider_ko_bounded_repair_evidence_is_complete_and_auditable() -> None:
    report = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    assert report["benchmark"] == {
        "name": "huggingface-KREW/spider-ko",
        "split": "validation",
        "question_language": "ko",
        "items": 1034,
        "comparison": "AEGIS execution_match on the official per-db SQLite database",
        "official_leaderboard": False,
    }
    assert report["model"]["name"] == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    assert report["model"]["adapter"] is None
    assert report["model"]["schema_style"] == "mschema"
    assert report["model"]["quantization"] == "NF4 4-bit"
    assert report["model"]["bounded_repair"] is True
    assert report["model"]["repair_max_attempts"] == 1

    result = report["result"]
    assert result["initial_correct"] == 393
    assert result["final_correct"] == 434
    assert result["gained_correct"] == 41
    assert result["regressed_correct"] == 0
    assert result["execution_failures_reduced"] == 96
    assert result["schema_reference_failures_reduced"] == 95
    assert result["final_execution_accuracy"] > result["initial_execution_accuracy"]

    repair = report["repair"]
    assert repair["attempted"] == 307
    assert repair["execution_recovered"] == 96
    assert repair["correct_after_repair"] == 41
    assert repair["repair_decision_uses_gold"] is False

    provenance = report["provenance"]
    assert provenance["git_sha"] == "f2b8ccacb164caf5ccb7061ec4d768e5bab92791"
    assert provenance["dataset_sha256"] == (
        "db802bab717deb2e16f2fa6cbb493712ac294515e68a69f1a243c5238cf99b52"
    )
    assert provenance["gpu"] == "Tesla T4"
    assert provenance["raw_report_bytes"] == 1_308_093
    assert provenance["portfolio_evidence_ready"] is True
    assert len(provenance["raw_report_sha256"]) == 64
