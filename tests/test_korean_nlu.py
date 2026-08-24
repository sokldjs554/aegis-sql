"""Korean normalisation — the layer that removes a whole class of LLM failures.

Every date in this schema is a 'YYYYMMDD' string, so "작년 하반기" has to become
('20250701','20251231') deterministically.  Getting this wrong is invisible in a
demo and catastrophic in a report, which is why it is pinned by tests rather than
left to the model.
"""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "text,expected",
    [
        ("작년 하반기 계약", ("20250701", "20251231")),
        ("올해 상반기 신계약", ("20260101", "20260630")),
        ("2025년 2분기", ("20250401", "20250630")),
        ("2024년 7월", ("20240701", "20240731")),
        ("2023년", ("20230101", "20231231")),
        ("지난달", ("20260701", "20260731")),
        ("이번달", ("20260801", "20260831")),
        ("최근 3개월", ("20260525", "20260824")),
        ("최근 7일", ("20260818", "20260824")),
        ("올해 들어", ("20260101", "20260824")),
        ("재작년", ("20240101", "20241231")),
        ("2025-03", ("20250301", "20250331")),
    ],
)
def test_date_expressions(normalizer, text, expected):
    nq = normalizer.normalize(text)
    ranges = [r for _, r in nq.entities.get("date_range", [])]
    assert expected in ranges, f"{text} → {ranges}"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("20만원 이상", 200_000),
        ("1억5천만원", 150_000_000),
        ("3.5억", 350_000_000),
        ("천만원", 10_000_000),
        ("100만", 1_000_000),
        ("2,000원", 2_000),
        ("500000원", 500_000),
    ],
)
def test_amount_parsing(normalizer, text, expected):
    nq = normalizer.normalize(text)
    values = [v for _, v in nq.entities.get("amount", [])]
    assert expected in values, f"{text} → {values}"


@pytest.mark.parametrize(
    "text,expected",
    [
        # An explicit range must stay one range.  Matching the two dates
        # independently yields BETWEEN start AND start, which runs, returns
        # almost nothing, and looks like a real answer.
        ("2025년 7월 1일부터 12월 31일까지 체결된 계약", ("20250701", "20251231")),
        ("2025년 7월부터 12월까지 신계약", ("20250701", "20251231")),
        ("2024년 3월 1일부터 2025년 2월 28일까지", ("20240301", "20250228")),
        ("2025-03-01부터 2025-03-31까지", ("20250301", "20250331")),
        ("2025.01.01 ~ 2025.06.30 계약", ("20250101", "20250630")),
        ("20250101부터 20250630까지", ("20250101", "20250630")),
    ],
)
def test_explicit_date_ranges_stay_ranges(normalizer, text, expected):
    nq = normalizer.normalize(text)
    ranges = [r for _, r in nq.entities.get("date_range", [])]
    assert expected in ranges, f"{text} → {ranges}"
    assert len(ranges) == 1, f"the range was split into {ranges}"


def test_particles_are_stripped_but_raw_kept(normalizer):
    nq = normalizer.normalize("계약자의 보험료가 얼마인가요")
    assert "계약자" in nq.tokens
    assert "보험료" in nq.tokens


@pytest.mark.parametrize(
    "text,cue",
    [
        ("계약 건수는 몇 건이야", "aggregate"),
        ("상위 5개 상품", "ranking"),
        ("월별 추이", "temporal"),
        ("20만원 이상인 계약", "comparison"),
        ("전체 평균보다 높은 계약", "nested"),
        ("채널별 비중", "ratio"),
    ],
)
def test_cue_detection(normalizer, text, cue):
    nq = normalizer.normalize(text)
    assert nq.entities["cues"].get(cue) is True, f"{text} → {nq.entities['cues']}"


def test_top_k_extraction(normalizer):
    assert normalizer.normalize("상위 5개 지점").entities.get("top_k") == 5
    assert normalizer.normalize("top 3 상품").entities.get("top_k") == 3


def test_group_by_hints(normalizer):
    hints = normalizer.normalize("지점별 월별 계약 건수").entities.get("group_by_hint", [])
    assert "지점" in hints and "월" in hints


def test_intent_classification(normalizer):
    assert normalizer.normalize("계약이 몇 건이야").intent == "count"
    assert normalizer.normalize("보험료 합계는").intent == "sum"
    assert normalizer.normalize("평균 월납보험료").intent == "avg"


def test_normalizer_is_pure_given_today(normalizer):
    """Same input, same output — no hidden clock reads inside normalize()."""
    a = normalizer.normalize("최근 3개월 청구")
    b = normalizer.normalize("최근 3개월 청구")
    assert a.entities["date_range"] == b.entities["date_range"]


def test_ambiguity_flags_vague_metric(schema, glossary, normalizer):
    from aegis_sql.nlu.ambiguity import AmbiguityDetector

    det = AmbiguityDetector(schema, [e.term for e in glossary.entries])
    report = det.detect(normalizer.normalize("설계사 실적 좀 보여줘"))
    assert report.is_ambiguous
    assert report.clarifying_question
    assert len(report.options) >= 2


def test_ambiguity_accepts_specific_question(schema, glossary, normalizer):
    from aegis_sql.nlu.ambiguity import AmbiguityDetector

    det = AmbiguityDetector(schema, [e.term for e in glossary.entries])
    q = "2025년 하반기에 체결된 계약 건수를 지점별로 알려줘"
    assert not det.detect(normalizer.normalize(q)).is_ambiguous
