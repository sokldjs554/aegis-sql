# Final Quality Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the remaining governance fail-open path and reconcile code/documentation with the measured AEGIS-SQL evidence before application materials are finalized.

**Architecture:** Keep query behavior unchanged while making governance fail closed in two layers: unclassified columns default to `forbidden` at runtime, and CI requires every demo schema column to have an explicit grade. Documentation fixes are evidence-only changes: no new performance claims and no retroactive provenance edits.

**Tech Stack:** Python 3.10+, Pydantic, sqlglot, YAML, pytest, GitHub Actions.

**Spec:** Current repository evidence plus the final application review checklist discussed on 2026-09-15.

> **Implementation review note:** The initial plan proposed a runtime startup exception for incomplete classification. Review found that the safer and simpler invariant is already available: unknown columns can fail closed as `forbidden`. The final implementation therefore combines that runtime fallback with a CI completeness test, so schema drift is blocked in CI without introducing unnecessary startup coupling.

## Global Constraints

- Do not change historical experiment SHA/provenance values.
- Do not claim bounded-repair Spider-KO gains before the 1,034-row GPU run exists.
- Keep existing query semantics and benchmark outputs unchanged.
- Use TDD/evidence checks for governance behavior; run the full CI matrix before merge.

---

### Task 1: Fail closed on missing column classifications

**Files:**
- Modify: `src/aegis_sql/config.py`
- Modify: `configs/default.yaml`
- Modify: `configs/policy/insurance.yaml`
- Create: `tests/test_policy_completeness.py`

**Final invariant:**
- Runtime fallback for an unclassified physical column is `Sensitivity.FORBIDDEN`.
- Every current demo-schema column is explicitly classified in `insurance.yaml`.
- CI fails if the schema adds a column without adding a policy grade.

- [x] **Step 1: Add a policy-completeness test**

The test compares every `SchemaGraph.all_columns` entry with the explicit policy keys.

- [x] **Step 2: Confirm the original RED state**

The first draft expected a runtime completeness helper that did not exist, and CI failed as expected. Review then simplified the design to avoid unnecessary startup coupling.

- [x] **Step 3: Make the runtime fallback fail closed**

Set both typed config and YAML/default config to `forbidden` for unclassified columns.

- [x] **Step 4: Explicitly grade every demo column**

Keep existing sensitive grades unchanged and add explicit `public` entries for every remaining column in `data/demo/schema.sql`.

- [x] **Step 5: Keep CI as the schema-drift gate**

`tests/test_policy_completeness.py` verifies explicit coverage and forbidden fallback behavior.

### Task 2: Reconcile router and sLLM documentation with measured implementation

**Files:**
- Modify: `src/aegis_sql/router/cascade.py`
- Modify: `docs/SLM.md`
- Modify: `.gitignore`

- [x] **Step 1: Replace stale router size**

Change `15M-parameter` to the shipped/measured `5.3M-parameter` checkpoint description and state that the SLM tier is default-disabled until promotion criteria are met.

- [x] **Step 2: Separate smoke duration from full training duration**

Distinguish the ~2-minute quick smoke path from the measured full 5.3M SFT+DPO run (~61 minutes).

- [x] **Step 3: Fix stale Make target comment**

Use `make train-slm`, not `make train`.

### Task 3: Reconcile README evidence language

**Files:**
- Modify: `README.md`

- [x] **Step 1: Clarify cascade terminology**

Describe the architecture as a four-tier ladder (`TEMPLATE → SLM → LLM → ENSEMBLE`), while the unpromoted SLM remains disabled by default.

- [x] **Step 2: Clarify the improvement loop**

Remove the ambiguous runtime “self-improving” tagline and define the loop as offline feedback/retraining. Disclose that the published 5.3M DPO checkpoint uses 900 synthetic preference pairs because production repair logs do not yet exist.

- [x] **Step 3: Separate historical full-scale evidence from current CI scale**

State that the saved full-scale template report used the full 373,778-row DB on Ubuntu/Python 3.13, while current PR CI deliberately builds a 0.25-scale 93,703-row DB and runs Python 3.10/3.11/3.12 for regression speed.

- [x] **Step 4: Keep current measured flywheel count**

Keep 12,416 pairs (train 9,914 / dev 1,192 / test 1,310) in current README descriptions; preserve older experiment-specific manifests as historical snapshots.

### Task 4: Verify and merge

- [x] **Step 1: Open PR and run CI**
- [x] **Step 2: Review the full diff for evidence/provenance regressions**
- [ ] **Step 3: Merge only after all six CI jobs pass**
- [ ] **Step 4: Verify post-merge `main` CI**
