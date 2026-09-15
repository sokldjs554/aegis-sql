from __future__ import annotations

import pytest

from aegis_sql.types import Sensitivity
from aegis_sql.verify.ast_guard import PolicyDocument, PolicyGuard


def test_policy_explicitly_classifies_every_demo_column(schema, guard):
    assert guard.policy.unclassified_columns(schema) == ()


def test_policy_guard_rejects_incomplete_classification(schema, settings):
    incomplete = PolicyDocument(
        columns={"TB_CUST.RRNO_ENC": Sensitivity.FORBIDDEN},
        strict_classification=True,
    )

    with pytest.raises(ValueError, match="unclassified schema columns"):
        PolicyGuard(schema, incomplete, settings)


def test_permissive_policy_keeps_classification_gate_disabled(schema, settings):
    policy = PolicyDocument.permissive()
    guard = PolicyGuard(schema, policy, settings)

    assert policy.strict_classification is False
    assert guard.policy is policy
