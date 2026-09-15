from __future__ import annotations

from aegis_sql.types import Sensitivity


def test_policy_explicitly_classifies_every_demo_column(schema, guard):
    classified = set(guard.policy.columns)
    schema_columns = {f"{col.table}.{col.name}".upper() for col in schema.all_columns}

    assert schema_columns <= classified


def test_policy_defaults_unknown_columns_to_forbidden(guard):
    assert guard.policy.default_sensitivity is Sensitivity.FORBIDDEN
    assert guard.policy.sensitivity("TB_NEW", "UNCLASSIFIED_COL") is Sensitivity.FORBIDDEN
