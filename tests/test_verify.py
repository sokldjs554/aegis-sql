"""Static checking, sandboxed execution, self-repair and execution voting."""

from __future__ import annotations

import pytest

from aegis_sql.types import SQLCandidate, Tier

# --------------------------------------------------------------------------- #
# executor
# --------------------------------------------------------------------------- #


def test_executor_returns_rows(executor):
    r = executor.execute("SELECT COUNT(*) AS c FROM TB_CTRT")
    assert r.ok and r.row_count == 1 and r.columns == ["c"]


def test_executor_is_read_only(executor):
    r = executor.execute("DELETE FROM TB_CTRT")
    assert not r.ok and r.error


def test_executor_reports_errors_without_raising(executor):
    r = executor.execute("SELECT nope FROM TB_CTRT")
    assert not r.ok and "nope" in (r.error or "")


def test_executor_truncates(demo_db):
    from aegis_sql.verify.executor import SQLExecutor

    ex = SQLExecutor(demo_db, timeout_s=5.0, max_rows=10)
    r = ex.execute("SELECT CTRT_NO FROM TB_CTRT")
    assert r.ok and r.row_count == 10 and r.truncated
    ex.close()


def test_explain_is_a_dry_run(executor):
    r = executor.explain("SELECT COUNT(*) FROM TB_CTRT WHERE CTRT_STAT_CD='01'")
    assert r.ok


def test_result_signature_is_order_insensitive():
    from aegis_sql.types import ExecutionResult

    a = ExecutionResult(ok=True, columns=["x"], rows=[(1,), (2,)], row_count=2)
    b = ExecutionResult(ok=True, columns=["x"], rows=[(2,), (1,)], row_count=2)
    assert a.result_signature() == b.result_signature()


# --------------------------------------------------------------------------- #
# static checks
# --------------------------------------------------------------------------- #


def test_static_check_catches_date_format_mismatch(schema, join_graph, profile):
    from aegis_sql.verify.static_check import StaticChecker

    checker = StaticChecker(schema, join_graph, profile)
    issues = checker.check("SELECT COUNT(*) FROM TB_CTRT WHERE CTRT_DT >= '2025-01-01'")
    assert any(v.code == "DATE_FORMAT_MISMATCH" for v in issues), [str(v) for v in issues]


def test_static_check_catches_date_function_on_text_column(schema, join_graph, profile):
    from aegis_sql.verify.static_check import StaticChecker

    checker = StaticChecker(schema, join_graph, profile)
    issues = checker.check("SELECT COUNT(*) FROM TB_CTRT WHERE DATE(CTRT_DT) > '2025-01-01'")
    assert issues


def test_static_check_catches_unknown_column(schema, join_graph, profile):
    from aegis_sql.verify.static_check import StaticChecker

    issues = StaticChecker(schema, join_graph, profile).check("SELECT CTRT_STATUS FROM TB_CTRT")
    assert any("UNKNOWN" in v.code for v in issues)


def test_static_check_accepts_valid_sql(schema, join_graph, profile):
    from aegis_sql.verify.static_check import StaticChecker

    sql = ("SELECT cd.CD_NM, COUNT(*) AS c FROM TB_CTRT t "
           "JOIN TB_COMM_CD cd ON cd.CD_GRP='CHNL' AND cd.CD=t.CHNL_CD GROUP BY cd.CD_NM")
    blocking = [v for v in StaticChecker(schema, join_graph, profile).check(sql) if v.severity in {"error", "block"}]
    assert not blocking, [str(v) for v in blocking]


# --------------------------------------------------------------------------- #
# repair
# --------------------------------------------------------------------------- #


@pytest.fixture()
def repairer(schema, profile, join_graph, executor):
    from aegis_sql.verify.repair import SelfRepairer
    from aegis_sql.verify.static_check import StaticChecker

    return SelfRepairer(
        schema=schema, profile=profile, join_graph=join_graph, executor=executor,
        static_checker=StaticChecker(schema, join_graph, profile), llm_repair=None, max_attempts=3,
    )


def _ctx(sql, error, question="테스트"):
    from aegis_sql.verify.repair import RepairContext

    return RepairContext(question=question, schema_card="", linked=None, sql=sql, error=error, attempt=0)


def test_repair_fixes_unknown_column(repairer, executor):
    sql = "SELECT COUNT(*) FROM TB_CTRT WHERE CTRT_STAT_CODE = '02'"
    err = executor.execute(sql).error or ""
    fixed, steps = repairer.repair(sql, err, _ctx(sql, err))
    assert fixed and executor.execute(fixed).ok
    assert any(s.strategy == "unknown-column" for s in steps)


def test_repair_fixes_date_literal_format(repairer, executor):
    sql = "SELECT COUNT(*) FROM TB_CTRT WHERE CTRT_DT BETWEEN '2025-07-01' AND '2025-12-31'"
    fixed, steps = repairer.repair(sql, "date format", _ctx(sql, "date format"))
    assert fixed is not None
    assert "20250701" in fixed
    result = executor.execute(fixed)
    assert result.ok and result.rows[0][0] > 0, "the repaired query must actually return data"


def test_repair_fixes_near_miss_table_name(repairer, executor):
    sql = "SELECT COUNT(*) FROM TB_CTRTS"
    err = executor.execute(sql).error or ""
    fixed, steps = repairer.repair(sql, err, _ctx(sql, err))
    assert fixed and executor.execute(fixed).ok
    assert steps and all(s.strategy for s in steps)


def test_repair_gives_up_gracefully(repairer, executor):
    """A hopeless statement must return (None, steps) rather than raise or loop."""
    sql = "SELECT COUNT(*) FROM ZZZZZZZZ_NOT_A_TABLE"
    err = executor.execute(sql).error or ""
    fixed, steps = repairer.repair(sql, err, _ctx(sql, err))
    assert fixed is None
    assert isinstance(steps, list)


# --------------------------------------------------------------------------- #
# self-consistency
# --------------------------------------------------------------------------- #


def test_vote_picks_the_majority_result(executor):
    from aegis_sql.verify.selfconsistency import vote

    same_a = SQLCandidate(sql="SELECT COUNT(*) AS c FROM TB_CTRT WHERE CTRT_STAT_CD = '02'", tier=Tier.LLM)
    same_b = SQLCandidate(sql="SELECT COUNT(CTRT_NO) AS c FROM TB_CTRT WHERE CTRT_STAT_CD = '02'", tier=Tier.LLM)
    other = SQLCandidate(sql="SELECT COUNT(*) AS c FROM TB_CTRT WHERE CTRT_STAT_CD = '03'", tier=Tier.LLM)
    winner, stats = vote([other, same_a, same_b], executor)
    assert winner is not None
    assert winner.sql in {same_a.sql, same_b.sql}
    assert stats["groups"] >= 2
    assert 0 < stats["agreement"] <= 1


def test_vote_marks_invalid_candidates(executor):
    from aegis_sql.verify.selfconsistency import vote

    good = SQLCandidate(sql="SELECT COUNT(*) FROM TB_CTRT")
    bad = SQLCandidate(sql="SELECT FROM WHERE")
    winner, _ = vote([bad, good], executor)
    assert winner is not None and winner.sql == good.sql
    assert bad.valid is False
