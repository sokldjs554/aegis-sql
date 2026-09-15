"""Contracts for the reproducible Spider-KO bounded-repair GPU runner."""

from __future__ import annotations

import json
from pathlib import Path

RUNNER = Path("scripts/run_spider_ko_bounded_repair_colab.sh")
NOTEBOOK = Path("notebooks/spider_ko_bounded_repair.ipynb")


def _notebook_text() -> tuple[dict, str]:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    return payload, text


def test_runner_pins_winning_schema_model_quantization_and_one_repair():
    text = RUNNER.read_text(encoding="utf-8")

    assert 'MODEL="${MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"' in text
    assert 'SCHEMA_STYLE="mschema"' in text
    assert "--schema-style \"$SCHEMA_STYLE\"" in text
    assert "--bounded-repair" in text
    assert "--load-4bit" in text
    assert "--adapter" not in text
    assert "EXPECTED_ITEMS=1034" in text


def test_runner_requires_persistent_archive_and_validates_complete_evidence():
    text = RUNNER.read_text(encoding="utf-8")

    assert "ARCHIVE_DIR" in text
    assert "full runs require ARCHIVE_DIR" in text
    assert "run-meta.json" in text
    assert "git rev-parse HEAD" in text
    assert "portfolio_evidence_ready" in text
    assert 'evaluation.get("bounded_repair") is True' in text
    assert 'evaluation.get("repair_max_attempts") == 1' in text
    assert 'evaluation.get("schema_style") == "mschema"' in text
    assert "initial_execution_accuracy" in text
    assert "execution_accuracy" in text
    assert "initial_schema_reference_failures" in text
    assert "schema_reference_failures" in text
    assert "schema_reference_failures_reduced" in text


def test_colab_notebook_checks_gpu_mounts_drive_runs_repair_and_prints_metrics():
    payload, text = _notebook_text()

    assert payload["nbformat"] == 4
    assert payload["metadata"]["accelerator"] == "GPU"
    assert "torch.cuda.is_available" in text
    assert "drive.mount" in text
    assert "/content/drive" in text
    assert "run_spider_ko_bounded_repair_colab.sh" in text
    assert "initial_execution_accuracy" in text
    assert "execution_accuracy" in text
    assert "schema_reference_failures_reduced" in text
    assert "repair" in text
    assert "공식 Spider leaderboard" in text
