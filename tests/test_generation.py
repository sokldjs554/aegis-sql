"""SQL normalisation utilities and the deterministic template generator.

The template tier is the reason this repository runs with no API key, so it is
tested against the live database: every question here must produce SQL that
actually executes.
"""

from __future__ import annotations

import pytest

from aegis_sql.generation.base import GenerationContext
from aegis_sql.types import Tier

# --------------------------------------------------------------------------- #
# skeleton utilities
# --------------------------------------------------------------------------- #


def test_normalize_sql_is_stable():
    from aegis_sql.generation.skeleton import normalize_sql

    assert normalize_sql("SELECT   a FROM t   WHERE b = 1") == normalize_sql("select a from t where b=1")


def test_sql_skeleton_erases_literals_and_identifiers():
    from aegis_sql.generation.skeleton import sql_skeleton

    a = sql_skeleton("SELECT COUNT(*) FROM TB_CTRT WHERE MON_PRM >= 200000")
    b = sql_skeleton("SELECT COUNT(*) FROM TB_CLM WHERE CLM_AMT >= 999")
    assert a == b, f"{a} != {b}"


def test_mask_question_hides_values():
    from aegis_sql.generation.skeleton import mask_question

    masked = mask_question("2025년 계약 중 20만원 이상")
    assert "2025" not in masked and "20만원" not in masked


def test_sql_components_and_difficulty():
    from aegis_sql.generation.skeleton import difficulty_of, sql_components

    easy = "SELECT COUNT(*) FROM TB_CTRT"
    hard = ("WITH x AS (SELECT CUST_ID, COUNT(*) c FROM TB_CTRT GROUP BY CUST_ID) "
            "SELECT AVG(c) FROM x WHERE c > (SELECT AVG(c) FROM x)")
    assert sql_components(easy)["n_joins"] == 0
    assert sql_components(hard)["has_subquery"]
    assert difficulty_of(easy) == "easy"
    assert difficulty_of(hard) == "hard"


# --------------------------------------------------------------------------- #
# template generator
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def template_generator(schema, profile, join_graph, glossary, settings):
    from aegis_sql.generation.template_generator import TemplateGenerator

    return TemplateGenerator(schema, profile, join_graph, glossary, settings)


def _generate(gen, linker, normalizer, card_builder, question):
    nq = normalizer.normalize(question)
    linked = linker.link(nq)
    ctx = GenerationContext(
        question=question, normalized=nq, linked=linked,
        schema_card=card_builder.render(linked, style="mschema"),
        today="20260824", dialect="sqlite",
    )
    return gen.generate(ctx)


@pytest.fixture(scope="module")
def card_builder(schema, profile, join_graph):
    from aegis_sql.schema.card import SchemaCardBuilder

    return SchemaCardBuilder(schema, profile, join_graph)


QUESTIONS = [
    "작년 하반기에 체결된 계약 건수",
    "실효된 계약의 채널별 건수를 많은 순으로",
    "월납보험료가 20만원 이상인 계약 수",
    "2025년 보험금 지급액 합계",
    "지점별 신계약 건수 상위 5개",
    "전체 계약은 몇 건인가요?",
    "부지급 처리된 청구 건수",
    "고객 등급별 평균 총가입금액",
]


@pytest.mark.parametrize("question", QUESTIONS)
def test_template_generator_produces_executable_sql(
    template_generator, linker, normalizer, card_builder, executor, question
):
    result = _generate(template_generator, linker, normalizer, card_builder, question)
    assert result.candidates, f"no candidate for: {question}"
    sql = result.candidates[0].sql
    execution = executor.execute(sql)
    assert execution.ok, f"{question}\n{sql}\n{execution.error}"
    assert result.candidates[0].tier is Tier.TEMPLATE


def test_template_generator_uses_glossary_hint(
    template_generator, linker, normalizer, card_builder
):
    sql = _generate(template_generator, linker, normalizer, card_builder,
                    "실효된 계약은 몇 건이야?").candidates[0].sql
    assert "'02'" in sql, sql


def test_template_generator_uses_yyyymmdd_literals(
    template_generator, linker, normalizer, card_builder
):
    sql = _generate(template_generator, linker, normalizer, card_builder,
                    "작년 하반기에 체결된 계약 건수").candidates[0].sql
    assert "20250701" in sql and "-" not in sql.split("WHERE")[-1].split("'")[1] if "'" in sql else True


def test_template_generator_never_raises(template_generator, linker, normalizer, card_builder):
    """Even nonsense must yield syntactically valid SQL rather than an exception."""
    result = _generate(template_generator, linker, normalizer, card_builder, "asdf 1234 ???")
    assert result.candidates and result.candidates[0].sql.strip().lower().startswith(("select", "with"))


def test_template_generator_is_free(template_generator, linker, normalizer, card_builder):
    result = _generate(template_generator, linker, normalizer, card_builder, "전체 계약 건수")
    assert result.cost_usd == 0.0


# --------------------------------------------------------------------------- #
# LLM tier (mock)
# --------------------------------------------------------------------------- #


def test_extract_sql_handles_messy_output():
    from aegis_sql.generation.llm_generator import extract_sql

    fenced = "설명입니다.\n```sql\nSELECT 1\n```"
    bare = "SELECT 2 FROM t;"
    multi = "```sql\nSELECT 3\n```\n중간설명\n```sql\nSELECT 4\n```"
    plain_fence = "```\nSELECT 5\n```"
    assert extract_sql(fenced).strip() == "SELECT 1"
    assert extract_sql(bare).strip() == "SELECT 2 FROM t"
    assert extract_sql(multi).strip() == "SELECT 4"
    assert extract_sql(plain_fence).strip() == "SELECT 5"
    assert extract_sql("no sql here") is None


def test_mock_llm_is_offline_and_deterministic(settings):
    from aegis_sql.llm.base import Message
    from aegis_sql.llm.mock import MockLLM

    llm = MockLLM({"계약": "SELECT COUNT(*) FROM TB_CTRT"})
    msgs = [Message("user", "【질문】\n전체 계약 건수")]
    a = llm.complete(msgs)
    b = llm.complete(msgs)
    assert a.text == b.text
    assert "```sql" in a.text
    assert a.cost_usd == 0.0
    assert len(llm.complete_n(msgs, 3)) == 3


def test_get_llm_client_falls_back_without_keys(settings, monkeypatch):
    from aegis_sql.llm.mock import MockLLM
    from aegis_sql.llm.providers import get_llm_client

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert isinstance(get_llm_client(settings), MockLLM)


def test_rejected_sampling_params_parses_the_real_error():
    """claude-sonnet-5 가 실제로 돌려준 400 문구에서 temperature 를 짚어내야 한다."""
    from aegis_sql.llm.providers import _rejected_sampling_params

    exc = Exception(
        "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
        "'message': '`temperature` is deprecated for this model.'}, 'request_id': 'req_x'}"
    )
    assert _rejected_sampling_params(exc) == {"temperature"}
    # 문구가 달라도(value-constraint 형태) invalid_request_error + 파라미터명이면 잡아야 한다 —
    # 특정 문구에만 반응하던 휴리스틱이 ensemble 경로의 400 을 놓친 실측 사고의 회귀.
    variant = Exception(
        "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
        "'message': '`temperature` may only be set to 1 for this model.'}}"
    )
    assert _rejected_sampling_params(variant) == {"temperature"}
    # 무관한 오류는 건드리지 않는다 — 401 을 sampling 문제로 오인하면 안 된다.
    auth = Exception(
        "Error code: 401 - {'type': 'error', 'error': {'type': 'authentication_error', "
        "'message': 'API key is invalid.'}}"
    )
    assert _rejected_sampling_params(auth) == set()


class _FakeMessage:
    content = "SELECT 1"
    response_metadata: dict = {}
    usage_metadata: dict = {}


def test_known_model_never_sends_sampling_params(monkeypatch):
    """claude-sonnet-5 는 temperature 를 받지 않는 모델로 알려져 있으므로,
    첫 400 왕복조차 없이 처음부터 보내지 않아야 한다."""
    from aegis_sql.config import get_settings
    from aegis_sql.llm.base import Message
    from aegis_sql.llm.providers import AnthropicClient

    calls: list[dict] = []

    class _FakeChat:
        def invoke(self, payload, **kwargs):
            calls.append(dict(kwargs))
            return _FakeMessage()

    client = AnthropicClient(get_settings(), model="claude-sonnet-5")
    monkeypatch.setattr(client, "available", lambda: True)
    monkeypatch.setattr(client, "_build", lambda: _FakeChat())

    # ensemble 경로가 넘기는 명시적 temperature 까지 포함해서 제거되어야 한다.
    result = client.complete([Message(role="user", content="ping")], temperature=0.7)
    assert result.text == "SELECT 1"
    assert len(calls) == 1 and "temperature" not in calls[0]
    assert "temperature" not in client._chat_kwargs()


def test_client_learns_rejection_and_shares_it_across_instances(monkeypatch):
    """시드 목록에 없는 모델이 temperature 를 거부하면: 빼고 즉시 1회 재시도하고,
    같은 모델의 다른 인스턴스(예: 사전 점검 → 엔진)도 그 학습을 물려받아야 한다."""
    from aegis_sql.config import get_settings
    from aegis_sql.llm.base import Message
    from aegis_sql.llm.providers import _MODEL_REJECTED_PARAMS, AnthropicClient

    model = "claude-test-learns-rejection"
    _MODEL_REJECTED_PARAMS.pop(("anthropic", model), None)  # 전역 레지스트리 격리

    calls: list[dict] = []

    class _FakeChat:
        def invoke(self, payload, **kwargs):
            calls.append(dict(kwargs))
            if "temperature" in kwargs:
                raise Exception(
                    "Error code: 400 - {'type': 'error', 'error': "
                    "{'type': 'invalid_request_error', "
                    "'message': '`temperature` may only be set to 1 for this model.'}}"
                )
            return _FakeMessage()

    first = AnthropicClient(get_settings(), model=model)
    monkeypatch.setattr(first, "available", lambda: True)
    monkeypatch.setattr(first, "_build", lambda: _FakeChat())
    assert first.complete([Message(role="user", content="ping")]).text == "SELECT 1"
    assert len(calls) == 2 and "temperature" in calls[0] and "temperature" not in calls[1]

    second = AnthropicClient(get_settings(), model=model)
    monkeypatch.setattr(second, "available", lambda: True)
    monkeypatch.setattr(second, "_build", lambda: _FakeChat())
    assert second.complete([Message(role="user", content="ping")], temperature=0.7).text == "SELECT 1"
    assert len(calls) == 3 and "temperature" not in calls[-1]

    _MODEL_REJECTED_PARAMS.pop(("anthropic", model), None)
