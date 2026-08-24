"""Hybrid schema linking, the business glossary and few-shot selection."""

from __future__ import annotations

import numpy as np
import pytest

from aegis_sql.retrieval.embedder import HashingEmbedder
from aegis_sql.retrieval.vectorstore import NumpyVectorStore


def test_hashing_embedder_is_deterministic_and_normalised():
    emb = HashingEmbedder(dim=128)
    a = emb.encode(["실효된 계약의 채널별 비중"])
    b = emb.encode(["실효된 계약의 채널별 비중"])
    assert np.allclose(a, b)
    assert pytest.approx(1.0, abs=1e-5) == float(np.linalg.norm(a[0]))


def test_hashing_embedder_ranks_related_text_higher():
    emb = HashingEmbedder(dim=256)
    q = emb.encode(["계약 상태 코드"])[0]
    near = emb.encode(["CTRT_STAT_CD 계약상태코드"])[0]
    far = emb.encode(["병원명 진단코드"])[0]
    assert float(q @ near) > float(q @ far)


def test_numpy_vector_store_roundtrip(tmp_path):
    store = NumpyVectorStore("t")
    vecs = np.eye(3, dtype=np.float32)
    store.add(["a", "b", "c"], ["A", "B", "C"], vecs, [{"k": "1"}, {"k": "2"}, {"k": "1"}])
    hits = store.search(vecs[0], k=2)
    assert hits[0][0] == "a"
    filtered = store.search(vecs[0], k=3, where={"k": "1"})
    assert {h[0] for h in filtered} == {"a", "c"}
    store.persist(tmp_path / "s")
    fresh = NumpyVectorStore("t")
    fresh.load(tmp_path / "s")
    assert len(fresh) == 3


def test_glossary_matches_aliases(glossary):
    hits = glossary.lookup(["유지율", "채널"], "채널별 유지율")
    assert any(e.term == "유지율" for e, _ in hits)


def test_glossary_prefers_specific_term(glossary):
    hits = glossary.lookup(["보유계약"], "보유계약 건수")
    terms = [e.term for e, _ in hits]
    assert "보유계약" in terms


def test_glossary_entries_have_usable_hints(glossary):
    with_hint = [e for e in glossary.entries if e.sql_hint]
    assert len(with_hint) >= 20
    for e in with_hint:
        assert e.tables, f"{e.term} has a SQL hint but no table binding"


@pytest.mark.parametrize(
    "question,must_contain",
    [
        ("작년 하반기에 체결된 계약 건수를 지점별로", {"TB_CTRT"}),
        ("실효된 계약의 채널별 비중은?", {"TB_CTRT"}),
        ("보험금 지급이 가장 오래 걸린 청구", {"TB_CLM"}),
        ("민원 유형별 평균 만족도", {"TB_CS_TCKT"}),
        ("설계사별 모집 계약 수", {"TB_CTRT", "TB_AGNT"}),
    ],
)
def test_schema_linking_finds_the_right_tables(linker, normalizer, question, must_contain):
    linked = linker.link(normalizer.normalize(question))
    assert must_contain <= set(linked.tables), f"{question} → {linked.tables}"


def test_schema_linking_prunes(linker, normalizer, schema):
    linked = linker.link(normalizer.normalize("실효된 계약은 몇 건이야?"))
    assert len(linked.tables) < len(schema.tables), "linking must actually prune"
    assert 0 < linked.coverage < 1


def test_linking_records_evidence(linker, normalizer):
    linked = linker.link(normalizer.normalize("실효된 계약의 채널별 비중"))
    assert linked.evidence
    sources = {e.source for e in linked.evidence}
    assert sources & {"dense", "lexical", "glossary", "value"}


def test_value_linking_hits_code_label(linker, normalizer):
    """'실효' is a code *label*, not a column name — value linking must find it."""
    linked = linker.link(normalizer.normalize("실효된 계약"))
    assert "TB_CTRT.CTRT_STAT_CD" in linked.columns


def test_primary_keys_always_survive_pruning(linker, normalizer, schema):
    linked = linker.link(normalizer.normalize("계약별 청구 금액"))
    for table in linked.tables:
        info = schema.table(table)
        for pk in info.primary_key:
            assert f"{table}.{pk}" in linked.columns or not info.primary_key


def test_fewshot_selection_is_diverse(settings):
    from aegis_sql.retrieval.embedder import get_embedder
    from aegis_sql.retrieval.fewshot import FewShotSelector
    from aegis_sql.types import FewShotExample

    examples = [
        FewShotExample(question=f"{n}년 계약 건수", sql=f"SELECT COUNT(*) FROM TB_CTRT WHERE substr(CTRT_DT,1,4)='{n}'")
        for n in range(2018, 2026)
    ] + [
        FewShotExample(question="채널별 계약 건수", sql="SELECT CHNL_CD, COUNT(*) FROM TB_CTRT GROUP BY CHNL_CD"),
        FewShotExample(question="지점별 청구 금액", sql="SELECT b.BRCH_NM, SUM(c.CLM_AMT) FROM TB_CLM c JOIN TB_CTRT t ON t.CTRT_NO=c.CTRT_NO JOIN TB_AGNT a ON a.AGNT_ID=t.AGNT_ID JOIN TB_BRCH b ON b.BRCH_CD=a.BRCH_CD GROUP BY b.BRCH_NM"),
    ]
    from aegis_sql.nlu.korean import KoreanNormalizer

    sel = FewShotSelector(examples, get_embedder(settings))
    picked = sel.select(KoreanNormalizer().normalize("2024년 계약 건수"), k=3, diversity=0.6)
    assert len(picked) == 3
    skeletons = {p.sql_skeleton or p.sql for p in picked}
    assert len(skeletons) >= 2, "MMR must not return three copies of one pattern"
