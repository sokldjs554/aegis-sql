"""Schema introspection, the FK join graph, profiling and prompt cards."""

from __future__ import annotations

import pytest

from aegis_sql.schema.card import SchemaCardBuilder, token_estimate
from aegis_sql.types import LinkedSchema


def test_all_tables_discovered(schema):
    expected = {
        "TB_COMM_CD", "TB_BRCH", "TB_AGNT", "TB_CUST", "TB_PROD",
        "TB_CTRT", "TB_CVRG", "TB_PAY", "TB_CLM", "TB_UW", "TB_CS_TCKT",
    }
    assert expected <= set(schema.tables)


def test_korean_data_dictionary_recovered_from_ddl(schema):
    """The Korean logical names live only in DDL comments — they must survive."""
    assert schema.table("TB_CTRT").comment == "계약"
    assert schema.column("TB_CTRT", "CTRT_STAT_CD").comment == "계약상태코드"
    assert schema.column("TB_CLM", "FRAUD_SCR").comment == "이상징후점수"
    commented = [c for c in schema.all_columns if c.comment]
    assert len(commented) > 90


def test_code_groups_inferred(schema):
    assert schema.column("TB_CTRT", "CTRT_STAT_CD").code_group == "CTRT_STAT"
    assert schema.column("TB_CUST", "VIP_GRD_CD").code_group == "VIP_GRD"
    assert schema.column("TB_CLM", "CLM_TYP_CD").code_group == "CLM_TYP"
    # A non-code column must not be mislabelled.
    assert schema.column("TB_CTRT", "CTRT_NO").code_group is None


def test_foreign_keys(schema):
    keys = {fk.key for fk in schema.foreign_keys}
    assert "TB_CTRT.CUST_ID->TB_CUST.CUST_ID" in keys
    assert "TB_CLM.CTRT_NO->TB_CTRT.CTRT_NO" in keys
    assert "TB_AGNT.BRCH_CD->TB_BRCH.BRCH_CD" in keys


def test_fingerprint_is_stable(schema):
    assert schema.fingerprint() == schema.fingerprint()
    assert len(schema.fingerprint()) == 16


def test_join_path_through_bridge_table(join_graph):
    """TB_CLM and TB_BRCH share no FK — the path must go CLM→CTRT→AGNT→BRCH."""
    path = join_graph.shortest_path("TB_CLM", "TB_BRCH")
    assert path is not None
    hops = [(e.left_table, e.right_table) for e in path]
    assert hops[0][0] == "TB_CLM"
    assert hops[-1][1] == "TB_BRCH"
    assert any("TB_AGNT" in h for h in hops)


def test_connect_adds_bridge_tables(join_graph):
    order, edges = join_graph.connect(["TB_CLM", "TB_BRCH", "TB_CUST"])
    assert {"TB_CTRT", "TB_AGNT"} <= set(order), "bridge tables must be pulled in"
    assert edges


def test_code_join_includes_group_predicate(join_graph):
    edge = join_graph.code_join("TB_CTRT", "CTRT_STAT_CD")
    sql = edge.to_sql("t", "cd")
    assert "cd.CD_GRP = 'CTRT_STAT'" in sql, "joining the code table without CD_GRP is a silent bug"
    assert "t.CTRT_STAT_CD = cd.CD" in sql


def test_profile_extracts_code_labels(profile):
    cp = profile.get("TB_CTRT", "CTRT_STAT_CD")
    assert cp.code_labels["02"] == "실효"
    assert cp.is_categorical


def test_profile_detects_yyyymmdd_columns(profile):
    assert profile.get("TB_CTRT", "CTRT_DT").is_yyyymmdd
    assert not profile.get("TB_CTRT", "MON_PRM").is_yyyymmdd


@pytest.mark.parametrize("style", ["mschema", "ddl", "compact", "slm"])
def test_card_styles_render(schema, profile, style):
    card = SchemaCardBuilder(schema, profile).render(style=style)
    assert "TB_CTRT" in card
    assert len(card) > 100


def test_pruned_card_is_much_smaller(schema, profile):
    builder = SchemaCardBuilder(schema, profile)
    full = token_estimate(builder.render(style="mschema"))
    linked = LinkedSchema(
        tables=["TB_CTRT", "TB_CUST"],
        columns=["TB_CTRT.CTRT_NO", "TB_CTRT.CTRT_STAT_CD", "TB_CUST.CUST_ID"],
    )
    pruned = token_estimate(builder.render(linked, style="mschema"))
    assert pruned * 3 < full, f"pruning saved too little: {full} → {pruned}"


def test_code_dictionary_present_in_card(schema, profile):
    linked = LinkedSchema(tables=["TB_CTRT"], columns=["TB_CTRT.CTRT_STAT_CD"])
    card = SchemaCardBuilder(schema, profile).render(linked, style="mschema")
    assert "CODE DICTIONARY" in card
    assert "'02'=실효" in card
