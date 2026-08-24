"""Governance — the tests that decide whether this engine could run in a bank.

These are written as adversarial probes, not happy paths: each one is a way a
model (or a user steering a model) could try to get data out, and each must be
stopped by the AST guard rather than by a request in the prompt.
"""

from __future__ import annotations

import pytest

from aegis_sql.types import Sensitivity


def _codes(verdict) -> set[str]:
    return {v.code for v in verdict.violations}


# --------------------------------------------------------------------------- #
# hard blocks
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT CUST_NM, RRNO_ENC FROM TB_CUST LIMIT 5",
        "SELECT c.RRNO_ENC FROM TB_CUST c",
        "SELECT * FROM TB_CUST",
        "SELECT x.r FROM (SELECT RRNO_ENC AS r FROM TB_CUST) x",
        "WITH q AS (SELECT RRNO_ENC FROM TB_CUST) SELECT * FROM q",
        "SELECT COUNT(*) FROM TB_CUST WHERE RRNO_ENC IS NOT NULL",
    ],
)
def test_forbidden_column_is_blocked_everywhere(guard, sql):
    verdict = guard.check(sql)
    assert not verdict.allowed, f"leaked: {sql}"
    assert "PII_FORBIDDEN" in _codes(verdict)


@pytest.mark.parametrize(
    "sql,code",
    [
        ("DELETE FROM TB_CTRT", "WRITE_FORBIDDEN"),
        ("UPDATE TB_CTRT SET CTRT_STAT_CD='01'", "WRITE_FORBIDDEN"),
        ("INSERT INTO TB_CUST VALUES (1)", "WRITE_FORBIDDEN"),
        ("DROP TABLE TB_CTRT", "WRITE_FORBIDDEN"),
        ("PRAGMA table_info(TB_CUST)", "PRAGMA_FORBIDDEN"),
        ("ATTACH DATABASE 'x.db' AS x", "ATTACH_FORBIDDEN"),
    ],
)
def test_non_select_statements_blocked(guard, sql, code):
    verdict = guard.check(sql)
    assert not verdict.allowed
    assert code in _codes(verdict)


def test_stacked_statement_blocked(guard):
    verdict = guard.check("SELECT 1; DELETE FROM TB_CTRT")
    assert not verdict.allowed


def test_masked_column_in_predicate_is_blocked(guard):
    """Filtering on a masked value reveals it one query at a time."""
    verdict = guard.check("SELECT CUST_ID FROM TB_CUST WHERE TELNO = '010-1234-5678'")
    assert not verdict.allowed
    assert "PII_PREDICATE" in _codes(verdict)


def test_masked_column_in_group_by_is_blocked(guard):
    verdict = guard.check("SELECT TELNO, COUNT(*) FROM TB_CUST GROUP BY TELNO")
    assert not verdict.allowed


def test_internal_column_row_level_is_blocked(guard):
    verdict = guard.check("SELECT CUST_ID, BRDT FROM TB_CUST LIMIT 10")
    assert not verdict.allowed
    assert "INTERNAL_ROWLEVEL" in _codes(verdict)


def test_internal_column_is_allowed_inside_aggregate(guard):
    verdict = guard.check(
        "SELECT AVG(CAST(substr(BRDT,1,4) AS INTEGER)) AS avg_birth_year FROM TB_CUST"
    )
    assert verdict.allowed, [str(v) for v in verdict.violations]


# --------------------------------------------------------------------------- #
# rewrites
# --------------------------------------------------------------------------- #


def test_masked_column_projection_is_rewritten(guard, executor):
    verdict = guard.check("SELECT CUST_ID, CUST_NM FROM TB_CUST LIMIT 3")
    assert verdict.allowed
    assert verdict.rewritten_sql and "substr" in verdict.rewritten_sql.lower()
    result = executor.execute(verdict.rewritten_sql)
    assert result.ok
    assert all("*" in str(row[1]) for row in result.rows), result.rows


def test_limit_is_injected(guard):
    verdict = guard.check("SELECT CTRT_NO FROM TB_CTRT")
    assert verdict.allowed
    assert "limit" in (verdict.rewritten_sql or "").lower()


def test_scalar_aggregate_is_not_limited(guard):
    verdict = guard.check("SELECT COUNT(*) FROM TB_CTRT")
    assert verdict.allowed
    assert "limit" not in (verdict.rewritten_sql or "").lower()


def test_oversized_limit_is_capped(guard, settings):
    verdict = guard.check("SELECT CTRT_NO FROM TB_CTRT LIMIT 100000")
    assert verdict.allowed
    assert str(settings.database.max_rows) in (verdict.rewritten_sql or "")


def test_row_policy_injected_from_context(guard, executor):
    verdict = guard.check("SELECT AGNT_ID, AGNT_NM FROM TB_AGNT", {"branch_cd": "BR003"})
    assert verdict.allowed
    assert "BR003" in (verdict.rewritten_sql or "")
    assert executor.execute(verdict.rewritten_sql).ok


def test_marketing_purpose_restricts_to_consenting_customers(guard):
    verdict = guard.check("SELECT CUST_ID FROM TB_CUST", {"purpose": "marketing"})
    assert verdict.allowed
    assert "MKT_AGR_YN" in (verdict.rewritten_sql or "")


def test_row_policy_absent_without_context(guard):
    verdict = guard.check("SELECT AGNT_ID FROM TB_AGNT")
    assert "BRCH_CD =" not in (verdict.rewritten_sql or "")


# --------------------------------------------------------------------------- #
# ordinary queries must survive
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT COUNT(*) FROM TB_CTRT WHERE CTRT_STAT_CD = '02'",
        "SELECT cd.CD_NM, COUNT(*) FROM TB_CTRT t JOIN TB_COMM_CD cd ON cd.CD_GRP='CHNL' AND cd.CD=t.CHNL_CD GROUP BY cd.CD_NM",
        "SELECT b.BRCH_NM, AVG(t.MON_PRM) FROM TB_CTRT t JOIN TB_AGNT a ON a.AGNT_ID=t.AGNT_ID JOIN TB_BRCH b ON b.BRCH_CD=a.BRCH_CD GROUP BY b.BRCH_NM",
        "WITH x AS (SELECT CTRT_NO, SUM(INSD_AMT) s FROM TB_CVRG GROUP BY CTRT_NO) SELECT COUNT(*) FROM x WHERE s > 0",
    ],
)
def test_legitimate_queries_pass_and_run(guard, executor, sql):
    verdict = guard.check(sql)
    assert verdict.allowed, [str(v) for v in verdict.violations]
    assert executor.execute(verdict.rewritten_sql or sql).ok


def test_policy_classification(guard):
    p = guard.policy
    assert p.sensitivity("TB_CUST", "RRNO_ENC") is Sensitivity.FORBIDDEN
    assert p.sensitivity("TB_CUST", "TELNO") is Sensitivity.MASKED
    assert p.sensitivity("TB_CUST", "BRDT") is Sensitivity.INTERNAL
    assert p.sensitivity("TB_CTRT", "MON_PRM") is Sensitivity.PUBLIC


def test_every_benchmark_governance_probe_is_refused(engine, benchmark):
    """End-to-end: the engine — not just the guard — must refuse these."""
    from aegis_sql.types import AnswerStatus

    leaked = []
    for item in benchmark:
        if item.expect != "blocked" or item.expected_violation == "MASK_APPLIED":
            continue
        bundle = engine.ask(item.question, allow_clarify=False)
        if bundle.status is not AnswerStatus.BLOCKED:
            leaked.append((item.id, bundle.status.value, bundle.sql))
    assert not leaked, f"governance probes not refused: {leaked}"
