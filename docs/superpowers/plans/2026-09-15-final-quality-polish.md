# Final Quality Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the remaining governance fail-open path and reconcile code/documentation with the measured AEGIS-SQL evidence before application materials are finalized.

**Architecture:** Keep runtime behavior unchanged except for one deliberate safety invariant: when governance is enabled, every physical schema column must have an explicit sensitivity grade. Preserve `PolicyDocument.permissive()` for policy-disabled test/dev flows. Documentation fixes are evidence-only changes: no new performance claims and no retroactive provenance edits.

**Tech Stack:** Python 3.10+, Pydantic, sqlglot, YAML, pytest, GitHub Actions.

**Spec:** Current repository evidence plus the final application review checklist discussed on 2026-09-15.

## Global Constraints

- Do not change historical experiment SHA/provenance values.
- Do not claim bounded-repair Spider-KO gains before the 1,034-row GPU run exists.
- Keep existing query semantics and benchmark outputs unchanged apart from stricter policy completeness validation.
- Use TDD for governance behavior; run the full CI matrix before merge.

---

### Task 1: Fail closed on missing column classifications

**Files:**
- Modify: `src/aegis_sql/verify/ast_guard.py`
- Modify: `src/aegis_sql/config.py`
- Modify: `configs/policy/insurance.yaml`
- Test: `tests/test_governance.py`

**Interfaces:**
- Consumes: `SchemaGraph.all_columns`, `PolicyDocument.columns`
- Produces: `PolicyDocument.unclassified_columns(schema) -> tuple[str, ...]`; `PolicyGuard` startup validation

- [ ] **Step 1: Write failing tests**

```python
def test_policy_explicitly_classifies_every_demo_column(schema, guard):
    assert guard.policy.unclassified_columns(schema) == ()


def test_policy_guard_rejects_incomplete_classification(schema, settings):
    from aegis_sql.verify.ast_guard import PolicyDocument, PolicyGuard
    incomplete = PolicyDocument(columns={"TB_CUST.RRNO_ENC": Sensitivity.FORBIDDEN})
    with pytest.raises(ValueError, match="unclassified schema columns"):
        PolicyGuard(schema, incomplete, settings)
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `pytest tests/test_governance.py -q`
Expected: missing `unclassified_columns` and/or no startup rejection.

- [ ] **Step 3: Implement minimal completeness gate**

Add `strict_classification: bool = True` to the parsed policy document and YAML. Add an `unclassified_columns` helper that compares `schema.all_columns` against explicit `columns` keys. In `PolicyGuard.__init__`, raise `ValueError` when strict mode is enabled and any physical column is missing. `PolicyDocument.permissive()` must set strict mode off.

- [ ] **Step 4: Explicitly grade every demo column**

Keep current sensitive grades unchanged and add explicit `public` entries for every remaining column in `data/demo/schema.sql`. Do not rely on a table wildcard because a newly added column must fail CI until someone classifies it.

- [ ] **Step 5: Run governance + full core tests**

Run: `pytest tests/test_governance.py -q` then `pytest -q -m "not slow" --maxfail=1`.
Expected: GREEN.

### Task 2: Reconcile router and sLLM documentation with measured implementation

**Files:**
- Modify: `src/aegis_sql/router/cascade.py`
- Modify: `docs/SLM.md`
- Modify: `.gitignore`

- [ ] **Step 1: Replace stale router size**

Change the cascade docstring from `15M-parameter` to the shipped/measured `5.3M-parameter` checkpoint description and note that the SLM tier is disabled by default until promotion criteria are met.

- [ ] **Step 2: Separate smoke duration from full training duration**

Change the opening SLM claim from “4-core CPU in a few minutes” to explicitly distinguish the ~2-minute quick smoke path from the measured full 5.3M run (~61 minutes including DPO).

- [ ] **Step 3: Fix stale Make target comment**

Use `make train-slm`, not `make train`.

### Task 3: Reconcile README evidence language

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Clarify cascade terminology**

Describe the architecture as a four-tier ladder (`TEMPLATE → SLM → LLM → ENSEMBLE`) while explaining that the default active path excludes the unpromoted SLM.

- [ ] **Step 2: Clarify self-improvement claim**

Define “자가개선” as an offline feedback/training loop, not runtime self-modification, and disclose that the published DPO checkpoint currently uses synthetic preference pairs because production repair logs do not yet exist.

- [ ] **Step 3: Separate historical full-scale evidence from current CI scale**

State that saved full-scale benchmark evidence uses the full demo DB, while PR CI deliberately builds a 0.25-scale DB for regression speed. Do not relabel one as the other.

- [ ] **Step 4: Keep current measured flywheel count**

Use 12,416 pairs (train 9,914 / dev 1,192 / test 1,310) wherever the current flywheel is described; preserve older experiment-specific manifests as historical snapshots.

### Task 4: Verify and merge

- [ ] **Step 1: Open PR and run CI**
- [ ] **Step 2: Review the full diff for evidence/provenance regressions**
- [ ] **Step 3: Merge only after all six CI jobs pass**
- [ ] **Step 4: Verify post-merge `main` CI**
