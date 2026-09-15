"""Contracts for the resumable Spider-KO schema ablation runner and Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path

RUNNER = Path("scripts/run_spider_ko_schema_ablation_colab.sh")
SUMMARIZER = Path("scripts/summarize_spider_ko_schema_ablation.py")
NOTEBOOK = Path("notebooks/spider_ko_schema_ablation.ipynb")


def _notebook_sources() -> list[str]:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(cell.get("source", [])) for cell in payload["cells"]]


def test_runner_holds_model_fixed_and_runs_exact_four_schema_styles():
    text = RUNNER.read_text(encoding="utf-8")

    assert 'MODEL="${MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"' in text
    assert 'STYLES=(slm ddl compact mschema)' in text
    assert '--schema-style "$style"' in text
    assert "--load-4bit" in text
    assert "--adapter" not in text
    assert "SMOKE" not in text


def test_runner_requires_persistent_storage_and_preserves_each_style_immediately():
    text = RUNNER.read_text(encoding="utf-8")

    assert "ARCHIVE_DIR" in text
    assert "full runs require ARCHIVE_DIR" in text
    assert 'REPORT="$RUN_DIR/spider-ko-$style.json"' in text
    assert "portfolio_evidence_ready" in text
    assert "continue" in text
    assert "FORCE" in text


def test_runner_pins_resume_to_one_git_sha_and_summarizes_only_complete_runs():
    text = RUNNER.read_text(encoding="utf-8")

    assert "git rev-parse HEAD" in text
    assert "run-meta.json" in text
    assert "git_sha" in text
    assert "summarize_spider_ko_schema_ablation.py" in text
    assert "spider-ko-schema-ablation-summary.json" in text


def test_summarizer_accepts_named_style_reports_and_expected_items():
    text = SUMMARIZER.read_text(encoding="utf-8")

    for option in ("--slm", "--ddl", "--compact", "--mschema"):
        assert option in text
    assert "--expected-items" in text
    assert "summarize_schema_ablation" in text
    assert "json.dumps" in text


def test_colab_notebook_checks_gpu_mounts_drive_runs_runner_and_prints_summary():
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = _notebook_sources()
    text = "\n".join(sources)

    assert payload["nbformat"] == 4
    assert payload["metadata"]["accelerator"] == "GPU"
    assert "torch.cuda.is_available" in text
    assert "drive.mount" in text
    assert "/content/drive" in text
    assert "run_spider_ko_schema_ablation_colab.sh" in text
    assert "spider-ko-schema-ablation-summary.json" in text
    assert "schema_reference_failures" in text
    assert "delta_accuracy_pp_vs_slm" in text
    assert "공식 Spider leaderboard" in text
