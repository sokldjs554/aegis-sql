from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from aegis_sql.schema.card import SchemaCardBuilder
from aegis_sql.training.hf_experiment import (
    build_qwen_summary,
    experiment_manifest,
    file_sha256,
    load_jsonl,
    prepare_records,
    records_sha256,
    select_records,
    user_prompt,
)

ROOT = Path(__file__).resolve().parents[1]


def test_prepare_records_reuses_real_schema_card(schema, profile, join_graph):
    builder = SchemaCardBuilder(schema, profile, join_graph)
    rows = [
        {
            "question": "실효된 계약은 몇 건이야?",
            "sql": "SELECT COUNT(*) FROM TB_CTRT WHERE CTRT_STAT_CD = '02'",
            "difficulty": "easy",
            "template_id": "count_filter",
            "source": "backtranslate",
            "tables": ["TB_CTRT"],
        }
    ]
    examples = prepare_records(rows, builder)
    assert len(examples) == 1
    ex = examples[0]
    assert "TB_CTRT(" in ex.schema_card
    assert "CTRT_STAT_CD" in ex.schema_card
    assert "TB_CUST(" not in ex.schema_card
    assert ex.sql == rows[0]["sql"]
    prompt = user_prompt(ex)
    assert "### 스키마" in prompt
    assert "### 질문" in prompt
    assert ex.question in prompt
    assert ex.sql not in prompt  # target SQL is not leaked into the user prompt


def test_jsonl_digest_and_manifest_are_reproducible(tmp_path):
    train = tmp_path / "train.jsonl"
    dev = tmp_path / "dev.jsonl"
    train.write_text(json.dumps({"question": "q1", "sql": "SELECT 1"}) + "\n", encoding="utf-8")
    dev.write_text(json.dumps({"question": "q2", "sql": "SELECT 2"}) + "\n", encoding="utf-8")

    assert load_jsonl(train)[0]["question"] == "q1"
    digest = file_sha256(train)
    assert digest == file_sha256(train)
    assert len(digest) == 64

    manifest = experiment_manifest(
        model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        train_path=train,
        dev_path=dev,
        train_count=1,
        dev_count=1,
        seed=42,
        schema_fingerprint="abc123",
        qlora=True,
        lora_r=16,
        lora_alpha=32,
    )
    assert manifest["dataset"]["train_sha256"] == digest
    assert manifest["adapter"]["method"] == "QLoRA"
    assert manifest["adapter"]["targets"] == ["q_proj", "k_proj", "v_proj", "o_proj"]


def test_training_subset_selection_is_seeded_and_hashed():
    rows = [{"question": f"q{i}", "sql": f"SELECT {i}"} for i in range(12)]
    first = select_records(rows, limit=5, seed=20260824, shuffle_before_limit=True)
    again = select_records(rows, limit=5, seed=20260824, shuffle_before_limit=True)
    different = select_records(rows, limit=5, seed=7, shuffle_before_limit=True)

    assert first == again
    assert first != different
    assert len(records_sha256(first)) == 64
    assert records_sha256(first) == records_sha256(again)


def test_committed_qwen_dataset_snapshot_materializes_with_verified_hashes(tmp_path):
    out = tmp_path / "dataset"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "qwen_dataset_snapshot.py"),
            "materialize",
            "--snapshot",
            str(ROOT / "data" / "research" / "qwen_flywheel_v1"),
            "--out",
            str(out),
        ],
        check=True,
    )
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["splits"]["train"]["count"] == 9000
    assert manifest["splits"]["dev"]["count"] == 1153
    assert len(load_jsonl(out / "train.jsonl")) == 9000


def _measured_report(adapter: str | None) -> dict:
    counts = (20, 15, 10) if adapter is None else (24, 18, 12)
    return {
        "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "adapter": adapter,
        "evaluation": {
            "git_sha": "abc123",
            "benchmark_sha256": "bench-sha",
            "database_sha256": "db-sha",
            "quantization": "NF4 4-bit",
        },
        "runtime": {"cuda_available": True, "gpu": {"name": "NVIDIA T4"}},
        "items": 90,
        "execution_accuracy": 0.5 if adapter is None else 0.6,
        "by_difficulty": {
            "easy": {"correct": counts[0], "total": 30, "ex": counts[0] / 30},
            "medium": {"correct": counts[1], "total": 30, "ex": counts[1] / 30},
            "hard": {"correct": counts[2], "total": 30, "ex": counts[2] / 30},
        },
        "latency_ms": {"p50": 100.0, "p95": 200.0},
        "max_cuda_memory_bytes": 4 * 1024**3,
    }


def test_qwen_summary_requires_complete_measured_gpu_evidence():
    manifest = {
        "status": "trained",
        "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "git_sha": "abc123",
        "database": {"sha256": "db-sha"},
        "dataset": {
            "train_count": 9000,
            "dev_count": 1153,
            "source_snapshot": {"name": "aegis-qwen-flywheel-v1"},
        },
        "runtime": {"cuda_available": True, "gpu": {"name": "NVIDIA T4"}},
        "training": {"wall_clock_seconds": 123.4},
        "max_cuda_memory_bytes": 6 * 1024**3,
    }
    summary = build_qwen_summary(
        manifest,
        _measured_report(None),
        _measured_report("adapter"),
        expected_items=90,
        run_kind="full",
    )

    assert summary["portfolio_evidence_ready"]
    assert summary["delta_ex_percentage_points"] == 10.0
    assert summary["training"]["peak_cuda_memory_gib"] == 6.0

    incomplete = _measured_report(None)
    incomplete["items"] = 10
    with pytest.raises(ValueError, match="expected 90"):
        build_qwen_summary(
            manifest,
            incomplete,
            _measured_report("adapter"),
            expected_items=90,
            run_kind="full",
        )
