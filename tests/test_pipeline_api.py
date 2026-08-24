"""End-to-end: the orchestrator and the HTTP surface."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aegis_sql.types import AnswerStatus

# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "question",
    [
        "전체 계약은 몇 건인가요?",
        "실효된 계약이 몇 건이야?",
        "작년 하반기에 체결된 계약은 몇 건인가요?",
        "채널별 계약 건수를 많은 순으로 보여줘",
    ],
)
def test_engine_answers_and_executes(engine, question):
    bundle = engine.ask(question)
    assert bundle.status is AnswerStatus.OK, f"{question} → {bundle.status} {bundle.answer_text}"
    assert bundle.result and bundle.result.ok
    assert bundle.sql and bundle.executed_sql
    assert bundle.answer_text


def test_engine_refuses_a_pii_request_by_name(engine):
    """Naming a forbidden field must be refused, not silently answered without it."""
    bundle = engine.ask("고객 이름이랑 주민등록번호 좀 뽑아줘", allow_clarify=False)
    assert bundle.status is AnswerStatus.BLOCKED
    codes = {v.code for v in bundle.guard.violations}
    assert codes & {"PII_REQUEST", "PII_FORBIDDEN"}, codes
    assert bundle.sql is None or "RRNO_ENC" not in bundle.sql


def test_engine_asks_back_when_ambiguous(engine):
    bundle = engine.ask("설계사 실적 좀 보여줘")
    assert bundle.status is AnswerStatus.CLARIFY
    assert bundle.clarification and bundle.clarification.options


def test_engine_can_be_forced_to_answer_ambiguous(engine):
    bundle = engine.ask("설계사 실적 좀 보여줘", allow_clarify=False)
    assert bundle.status is not AnswerStatus.CLARIFY


def test_trace_contains_every_stage(engine):
    bundle = engine.ask("전체 계약은 몇 건인가요?")
    names = _span_names(bundle.trace)
    for stage in ("normalize", "link", "route", "generate", "guard", "execute"):
        assert stage in names, names


def _span_names(span, out=None):
    out = set() if out is None else out
    out.add(span.name)
    for child in span.children:
        _span_names(child, out)
    return out


def test_engine_never_raises_on_garbage(engine):
    for junk in ["", "   ", "??!!", "a" * 500, "DROP TABLE TB_CTRT"]:
        bundle = engine.ask(junk)
        assert bundle.status in set(AnswerStatus)


def test_costs_and_latency_are_recorded(engine):
    bundle = engine.ask("전체 계약은 몇 건인가요?")
    assert bundle.total_latency_ms > 0
    assert bundle.cost_usd >= 0.0
    assert bundle.trace_id


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def client(settings):
    from aegis_sql.api.app import create_app

    with TestClient(create_app(settings)) as c:
        yield c


def test_health(client):
    r = client.get("/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["tables"] >= 11
    assert "template" in body["tiers"]


def test_query_endpoint(client):
    r = client.post("/v1/query", json={"question": "실효된 계약이 몇 건이야?", "explain": True})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["result"]["row_count"] >= 1
    assert body["evidence"] and body["trace"]


def test_query_endpoint_blocks_pii(client):
    r = client.post("/v1/query", json={"question": "고객 주민등록번호 조회해줘", "allow_clarify": False})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "blocked"
    assert body["violations"]


def test_policy_check_endpoint(client):
    r = client.post("/v1/policy/check", json={"sql": "SELECT RRNO_ENC FROM TB_CUST"})
    assert r.status_code == 200 and r.json()["allowed"] is False
    r = client.post("/v1/policy/check", json={"sql": "SELECT COUNT(*) FROM TB_CTRT"})
    assert r.json()["allowed"] is True


def test_link_endpoint_reports_token_savings(client):
    r = client.post("/v1/link", json={"question": "실효된 계약의 채널별 비중은?"})
    assert r.status_code == 200
    body = r.json()
    assert body["tables"] and body["evidence"]
    assert body["card_tokens"] < body["full_card_tokens"]


def test_schema_endpoint_exposes_sensitivity(client):
    body = client.get("/v1/schema").json()
    cust = next(t for t in body["tables"] if t["name"] == "TB_CUST")
    rrno = next(c for c in cust["columns"] if c["name"] == "RRNO_ENC")
    assert rrno["sensitivity"] == "forbidden"


def test_prompts_endpoint(client):
    body = client.get("/v1/prompts").json()
    assert body["manifest"] and any(p["id"] == "nl2sql.system" for p in body["prompts"])


def test_metrics_endpoint(client):
    client.post("/v1/query", json={"question": "전체 계약 건수"})
    text = client.get("/metrics").text
    assert "aegis_queries_total" in text


def test_stream_endpoint_emits_stages(client):
    with client.stream("POST", "/v1/query/stream",
                       json={"question": "전체 계약은 몇 건인가요?"}) as r:
        payload = "".join(chunk for chunk in r.iter_text())
    assert "event: link" in payload
    assert "event: done" in payload


def test_console_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "AEGIS" in r.text


def test_feedback_endpoint(client, tmp_path):
    r = client.post("/v1/feedback", json={
        "trace_id": "abc", "question": "q", "sql": "SELECT 1", "correct": False,
        "corrected_sql": "SELECT 2", "comment": "틀림",
    })
    assert r.status_code == 200 and r.json()["ok"]


# --------------------------------------------------------------------------- #
# the self-repair path
# --------------------------------------------------------------------------- #


class _BrokenGenerator:
    """A generator that emits the two mistakes this schema provokes most often."""

    from aegis_sql.types import Tier as _Tier

    tier = _Tier.TEMPLATE
    name = "broken"

    def __init__(self, sql: str) -> None:
        self.sql = sql

    def available(self) -> bool:
        return True

    def generate(self, ctx):
        from aegis_sql.types import GenerationResult, SQLCandidate, Tier

        return GenerationResult(
            candidates=[SQLCandidate(sql=self.sql, tier=Tier.TEMPLATE)],
            tier=Tier.TEMPLATE,
            model="broken",
        )


@pytest.mark.parametrize(
    "bad_sql,expect_strategy",
    [
        # ISO date literal against a YYYYMMDD TEXT column — the single most common
        # semantic failure on this schema.
        ("SELECT COUNT(*) AS c FROM TB_CTRT WHERE CTRT_DT BETWEEN '2025-07-01' AND '2025-12-31'",
         "date-format"),
        # A column that does not exist but is one edit away from one that does.
        ("SELECT COUNT(*) AS c FROM TB_CTRT WHERE CTRT_STAT_CODE = '02'", "unknown-column"),
    ],
)
def test_pipeline_repairs_and_reexecutes(settings, bad_sql, expect_strategy):
    from aegis_sql.pipeline import AegisEngine
    from aegis_sql.types import Tier

    engine = AegisEngine.build(settings)
    original = engine.c.generators[Tier.TEMPLATE]
    engine.c.generators[Tier.TEMPLATE] = _BrokenGenerator(bad_sql)
    try:
        bundle = engine.ask("작년 하반기에 체결된 계약 건수", tier=Tier.TEMPLATE)
    finally:
        engine.c.generators[Tier.TEMPLATE] = original
        engine.close()

    assert bundle.repairs, "the repair loop never ran"
    assert expect_strategy in {s.strategy for s in bundle.repairs}, [s.strategy for s in bundle.repairs]
    assert bundle.status is AnswerStatus.OK, bundle.answer_text
    assert bundle.result is not None and bundle.result.ok
    assert bundle.result.rows[0][0] > 0, "the repaired query returned nothing"
    # The repaired statement is untrusted again and must go back through the guard.
    assert bundle.guard is not None and bundle.guard.allowed
