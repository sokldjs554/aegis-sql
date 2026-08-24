"""Settings layering and the versioned prompt registry."""

from __future__ import annotations

import pytest

from aegis_sql.config import Settings
from aegis_sql.prompts.registry import PromptRegistry


def test_defaults_load():
    s = Settings.load()
    assert s.retrieval.top_k_columns > 0
    assert s.verify.max_repair_attempts >= 1


def test_env_override(monkeypatch):
    monkeypatch.setenv("AEGIS_ROUTER__ESCALATE_THRESHOLD", "0.81")
    monkeypatch.setenv("AEGIS_LOG_JSON", "true")
    s = Settings.load()
    assert s.router.escalate_threshold == pytest.approx(0.81)
    assert s.log_json is True


def test_explicit_override_wins(monkeypatch):
    monkeypatch.setenv("AEGIS_GENERATION__MODEL", "from-env")
    s = Settings.load(generation={"model": "explicit"})
    assert s.generation.model == "explicit"


def test_prompt_registry_loads_and_hashes():
    reg = PromptRegistry.load("default")
    assert len(reg) >= 8
    rec = reg.get("nl2sql.system")
    assert rec.ref.startswith("nl2sql.system@")
    assert len(rec.hash) == 12


def test_prompt_render_enforces_declared_variables():
    reg = PromptRegistry.load("default")
    with pytest.raises(KeyError):
        reg.render("nl2sql.user", question="only one variable")


def test_prompt_render_tolerates_optional_blocks():
    reg = PromptRegistry.load("default")
    text = reg.render("nl2sql.user", schema_card="CARD", question="Q")
    assert "CARD" in text and "Q" in text


def test_system_prompt_states_the_date_rule():
    """The single highest-value rule for this schema must not silently vanish."""
    text = PromptRegistry.load("default").render("nl2sql.system", dialect="SQLite", today="20260824")
    assert "YYYYMMDD" in text
    assert "20260824" in text


def test_override_creates_experiment_variant():
    reg = PromptRegistry.load("default")
    exp = reg.with_override("nl2sql.user", "{{ schema_card }} / {{ question }} / EXP")
    assert exp.get("nl2sql.user").hash != reg.get("nl2sql.user").hash
    assert "experiment" in exp.get("nl2sql.user").tags
    assert reg.get("nl2sql.user").template != exp.get("nl2sql.user").template


def test_manifest_covers_every_prompt():
    reg = PromptRegistry.load("default")
    assert set(reg.manifest()) == {r.id for r in reg}
