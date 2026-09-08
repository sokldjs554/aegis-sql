from __future__ import annotations

import json

from aegis_sql.schema.card import SchemaCardBuilder
from aegis_sql.training.hf_experiment import (
    experiment_manifest,
    file_sha256,
    load_jsonl,
    prepare_records,
    user_prompt,
)


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
