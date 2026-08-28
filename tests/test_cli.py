"""CLI smoke tests.

The CLI is the surface a reviewer touches first, so a broken command is a broken
first impression.  These run the real Typer app in-process against the real demo
database — no mocking — and assert on exit codes and the parts of the output that
carry meaning.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from aegis_sql.cli import app

runner = CliRunner()


@pytest.fixture(scope="module", autouse=True)
def _template_tier(demo_db):
    """Pin the CLI to the offline tier so these tests never need an API key."""
    import os

    os.environ["AEGIS_GENERATION__PROVIDER"] = "template"
    os.environ["AEGIS_DATABASE__PATH"] = str(demo_db)
    from aegis_sql.config import reset_settings_cache

    reset_settings_cache()
    yield
    os.environ.pop("AEGIS_GENERATION__PROVIDER", None)
    os.environ.pop("AEGIS_DATABASE__PATH", None)
    reset_settings_cache()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("ask", "demo", "eval", "serve", "link", "policy", "schema"):
        assert command in result.output


def test_ask_returns_sql_and_rows():
    result = runner.invoke(app, ["ask", "실효된 계약이 몇 건이야?"])
    assert result.exit_code == 0, result.output
    assert "SELECT" in result.output.upper()


def test_ask_json_is_parseable():
    result = runner.invoke(app, ["ask", "전체 계약은 몇 건인가요?", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] in {"ok", "clarify", "blocked", "failed"}
    assert payload["trace_id"]


def test_ask_explain_shows_evidence_and_trace():
    result = runner.invoke(app, ["ask", "채널별 계약 건수", "--explain"])
    assert result.exit_code == 0
    assert "스키마 링킹" in result.output
    assert "트레이스" in result.output


def test_ask_blocks_pii():
    result = runner.invoke(app, ["ask", "고객 주민등록번호 조회해줘"])
    assert result.exit_code == 0
    assert "차단" in result.output or "blocked" in result.output.lower()


def test_policy_command():
    ok = runner.invoke(app, ["policy", "SELECT COUNT(*) FROM TB_CTRT"])
    assert ok.exit_code == 0 and "허용" in ok.output
    bad = runner.invoke(app, ["policy", "SELECT RRNO_ENC FROM TB_CUST"])
    assert bad.exit_code == 0 and "차단" in bad.output


def test_link_command_shows_scores():
    result = runner.invoke(app, ["link", "실효된 계약의 채널별 비중은?"])
    assert result.exit_code == 0
    assert "TB_CTRT" in result.output


def test_schema_command_renders_card():
    result = runner.invoke(app, ["schema", "--style", "compact"])
    assert result.exit_code == 0 and "TB_CTRT" in result.output


def test_prompt_list_and_show():
    listing = runner.invoke(app, ["prompt", "list"])
    assert listing.exit_code == 0 and "nl2sql.system" in listing.output
    shown = runner.invoke(app, ["prompt", "show", "nl2sql.user"])
    assert shown.exit_code == 0 and "질문" in shown.output


def test_version_reports_tiers():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0 and "aegis-sql" in result.output


def test_demo_runs_all_paths():
    """The demo deliberately includes a PII refusal and an ambiguous question."""
    result = runner.invoke(app, ["demo", "--limit", "5"])
    assert result.exit_code == 0
    assert "차단" in result.output
    assert "되묻기" in result.output


@pytest.mark.slow
def test_eval_quick_produces_a_report(tmp_path):
    report = tmp_path / "eval.md"
    result = runner.invoke(app, ["eval", "--limit", "8", "--report", str(report)])
    assert result.exit_code == 0, result.output
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "실행 정확도" in text and "거버넌스" in text
    assert (tmp_path / "eval.json").exists()


def test_eval_forced_llm_tier_without_provider_aborts_before_running():
    """LLM 프로바이더 없이 ``--tier llm`` 을 강제하면 문항을 하나도 돌리기 전에
    명확한 이유와 함께 중단해야 한다 — 조용한 template 폴백 리포트 방지."""
    result = runner.invoke(app, ["eval", "--tier", "llm", "--limit", "1"])
    assert result.exit_code == 1
    assert "LLM" in result.output
    assert "평가 시작" not in result.output
