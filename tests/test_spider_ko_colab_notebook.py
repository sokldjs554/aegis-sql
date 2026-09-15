"""Contract tests for the one-click Spider-KO Colab full-run notebook."""

import json
from pathlib import Path


NOTEBOOK = Path("notebooks/spider_ko_full_run.ipynb")


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _sources(nb: dict) -> list[str]:
    return ["".join(cell.get("source", [])) for cell in nb["cells"]]


def test_notebook_is_valid_colab_ipynb_with_gpu_accelerator():
    nb = _notebook()

    assert nb["nbformat"] == 4
    assert nb["metadata"]["kernelspec"]["name"] == "python3"
    assert nb["metadata"]["accelerator"] == "GPU"
    assert nb["metadata"]["colab"]["name"] == "AEGIS-SQL Spider-KO Full Run"


def test_notebook_checks_cuda_before_starting_full_run():
    sources = _sources(_notebook())
    gpu_cell = next(source for source in sources if "torch.cuda.is_available" in source)

    assert "nvidia-smi" in gpu_cell
    assert "RuntimeError" in gpu_cell
    assert "GPU" in gpu_cell


def test_notebook_mounts_drive_and_uses_persistent_archive_dir():
    sources = _sources(_notebook())
    drive_cell = next(source for source in sources if "drive.mount" in source)

    assert "/content/drive" in drive_cell
    assert "AEGIS-SQL/spider-ko" in drive_cell
    assert "mkdir" in drive_cell or "mkdir(" in drive_cell


def test_notebook_clones_or_updates_main_and_runs_full_runner():
    sources = _sources(_notebook())
    repo_cell = next(source for source in sources if "sokldjs554/aegis-sql" in source)
    run_cell = next(source for source in sources if "run_spider_ko_colab.sh" in source)

    assert "git clone" in repo_cell
    assert "git pull" in repo_cell
    assert "main" in repo_cell
    assert "ARCHIVE_DIR" in run_cell
    assert "SMOKE" not in run_cell
    assert "bash scripts/run_spider_ko_colab.sh" in run_cell


def test_notebook_verifies_and_summarizes_archived_result():
    sources = _sources(_notebook())
    result_cell = next(source for source in sources if "portfolio_evidence_ready" in source)

    assert "spider-ko-full-" in result_cell
    assert "execution_accuracy" in result_cell
    assert "latency_ms" in result_cell
    assert "execution_failures" in result_cell
    assert "items" in result_cell
    assert "1034" in result_cell
    assert "raise" in result_cell


def test_notebook_warns_that_score_is_not_official_spider_leaderboard():
    text = "\n".join(_sources(_notebook()))

    assert "공식 Spider leaderboard" in text
    assert "AEGIS execution_match" in text
