"""Difficulty features — the evidence the cascade router routes on.

Routing is not a classification of the *question*; it is a prediction about a
*generator's own failure*.  The only thing worth predicting before a token is
spent is: "will the cheap tier get **this** question right?"  That prediction has
to be made from signals that are already on the table after normalisation and
schema linking, because anything that requires a model call has already spent
the money the router exists to save.

Three families of signal survive that constraint, and they are weighted very
differently on purpose:

1. **Question-side shape cues** (``has_nested_cue``, ``has_ratio_cue``, ...).
   These are the strongest evidence available, because they describe the *SQL
   shape* the answer must have.  ``"전체 평균보다 많은"`` means a correlated
   subquery or a window function — the single construct that separates what a
   15M-parameter in-house model can emit from what it cannot.  ``"비중"``
   forces a ratio over a conditional aggregate.  A question carrying neither is
   almost always a filter-plus-GROUP BY that the deterministic template grammar
   already covers.

2. **Linked sub-schema size** (``n_linked_tables``, ``max_join_depth``, ...).
   Deliberately *damped*.  The linker is recall-biased by construction — it
   honours ``retrieval.min_tables`` and expands one FK hop past every seed — so
   a trivial question like ``"전체 계약 건수"`` still comes back with six tables
   and a join depth of two.  Treating that as difficulty would route every easy
   question to a frontier model, which is exactly the failure this subsystem
   exists to prevent.  It is real signal, but noisy signal.

3. **Mapping uncertainty** (``ambiguity_score``, ``link_score_gap``,
   ``link_score_mean``, ``schema_coverage``).  ``link_score_gap`` — the margin
   between the best and second-best linked object — is the cheap proxy for "the
   retriever is not sure which column you meant".  A narrow gap is where a small
   model silently picks the wrong column and returns a plausible wrong number,
   the one failure mode execution checks cannot catch.

:func:`extract_features` is pure: same inputs, same vector, no clock, no RNG.
That is what lets the same function serve the online path and the offline
training set builder without a train/serve skew.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from aegis_sql.observability.logging import get_logger
from aegis_sql.types import AmbiguityReport, DifficultyFeatures, LinkedSchema, NormalizedQuestion

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps import cost at zero
    from aegis_sql.schema.graph import JoinGraph

log = get_logger("router.features")

#: ``nlu.korean.CUE_NAMES`` → the ``DifficultyFeatures`` flag it fills.
_CUE_TO_FIELD: dict[str, str] = {
    "aggregate": "has_aggregate_cue",
    "ranking": "has_ranking_cue",
    "temporal": "has_temporal_cue",
    "comparison": "has_comparison_cue",
    "nested": "has_nested_cue",
    "ratio": "has_ratio_cue",
    "set_op": "has_set_op_cue",
}

#: A link-score margin at or above this is treated as "the retriever is certain".
LINK_GAP_SATURATION = 0.30
#: Hybrid link scores are ~[0, 1.3] (a glossary hit can push a column past 1.0).
LINK_MEAN_SATURATION = 1.0


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


def extract_features(
    nq: NormalizedQuestion,
    linked: LinkedSchema | None,
    join_graph: JoinGraph | None,
    ambiguity: AmbiguityReport | None,
) -> DifficultyFeatures:
    """Build the router's input vector from the stages that already ran.

    Every argument except ``nq`` is optional so that the function stays usable
    from the flywheel and the eval harness, where linking may be skipped.
    """
    f = DifficultyFeatures()
    f.n_tokens = len(nq.tokens)
    f.n_value_candidates = len(nq.value_candidates)

    cues: Any = nq.entities.get("cues") or {}
    if isinstance(cues, dict):
        for cue, field_name in _CUE_TO_FIELD.items():
            setattr(f, field_name, int(bool(cues.get(cue))))

    if linked is not None:
        f.n_linked_tables = len(linked.tables)
        f.n_linked_columns = len(linked.columns)
        f.glossary_hits = len(linked.glossary)
        f.schema_coverage = float(linked.coverage)
        mean, gap = _evidence_stats(linked)
        f.link_score_mean = mean
        f.link_score_gap = gap
        if join_graph is not None and linked.tables:
            f.max_join_depth = join_graph.max_depth(linked.tables)

    if ambiguity is not None:
        f.ambiguity_score = float(ambiguity.score)

    return f


def _evidence_stats(linked: LinkedSchema) -> tuple[float, float]:
    """Mean evidence score and the top1−top2 margin.

    The margin is computed over the *sorted* scores rather than trusting the
    linker's ordering, so this stays correct if evidence is ever merged from
    two retrievers.
    """
    scores = sorted((float(e.score) for e in linked.evidence), reverse=True)
    if not scores:
        return 0.0, 0.0
    mean = sum(scores) / len(scores)
    gap = scores[0] - scores[1] if len(scores) > 1 else scores[0]
    return round(mean, 6), round(max(0.0, gap), 6)


# --------------------------------------------------------------------------- #
# Transparent fallback scorer
# --------------------------------------------------------------------------- #

#: Weight of every term in :func:`heuristic_difficulty`.  The positive weights
#: sum to exactly 1.0, so the raw score is already a probability-shaped number
#: and the table below *is* the model — there is nothing else to inspect.
#:
#: ==========================  ======  ==================================================
#: term                        weight  rationale
#: ==========================  ======  ==================================================
#: has_nested_cue               0.25   correlated subquery / window fn — the #1 SLM killer
#: has_ratio_cue                0.12   ratio over a conditional aggregate
#: has_set_op_cue               0.07   EXCEPT / OR-of-predicates restructuring
#: has_ranking_cue              0.05   ORDER BY + LIMIT, usually mechanical
#: has_comparison_cue           0.02   a literal predicate, already resolved by nlu.korean
#: has_temporal_cue             0.01   dates are pre-resolved to YYYYMMDD by nlu.korean
#: has_aggregate_cue            0.01   near-universal in analytics questions ⇒ low signal
#: max_join_depth               0.09   min(d,3)/3 — real cost, but linker-inflated
#: n_linked_tables              0.06   clip((n-2)/4) — the min_tables guard floors this at 2
#: n_linked_columns             0.03   min(n,40)/40 — prompt size, not logic complexity
#: schema_coverage              0.02   high coverage ⇒ pruning failed to narrow anything
#: ambiguity_score              0.10   the NLU layer's own "I am not sure" number
#: link_score_gap               0.09   deficit: 1 − min(gap/0.30, 1) — a tie is dangerous
#: link_score_mean              0.05   deficit: 1 − min(mean/1.0, 1) — weak evidence overall
#: n_tokens                     0.02   min(n,28)/28 — length is a weak proxy at best
#: n_value_candidates           0.01   min(n,5)/5 — more literals to bind correctly
#: glossary_hits               −0.05   −min(h,3)/3: a term-dictionary hit *removes* doubt
#: ==========================  ======  ==================================================
HEURISTIC_WEIGHTS: dict[str, float] = {
    "has_nested_cue": 0.25,
    "has_ratio_cue": 0.12,
    "has_set_op_cue": 0.07,
    "has_ranking_cue": 0.05,
    "has_comparison_cue": 0.02,
    "has_temporal_cue": 0.01,
    "has_aggregate_cue": 0.01,
    "max_join_depth": 0.09,
    "n_linked_tables": 0.06,
    "n_linked_columns": 0.03,
    "schema_coverage": 0.02,
    "ambiguity_score": 0.10,
    "link_score_gap": 0.09,
    "link_score_mean": 0.05,
    "n_tokens": 0.02,
    "n_value_candidates": 0.01,
    "glossary_hits": -0.05,
}

#: Each term is mapped into [0, 1] before it is weighted.  Two of them are
#: *deficits* — the router is made more nervous by the **absence** of linking
#: evidence, not by its presence.
_NORMALIZERS: dict[str, Callable[[DifficultyFeatures], float]] = {
    "has_nested_cue": lambda f: float(f.has_nested_cue),
    "has_ratio_cue": lambda f: float(f.has_ratio_cue),
    "has_set_op_cue": lambda f: float(f.has_set_op_cue),
    "has_ranking_cue": lambda f: float(f.has_ranking_cue),
    "has_comparison_cue": lambda f: float(f.has_comparison_cue),
    "has_temporal_cue": lambda f: float(f.has_temporal_cue),
    "has_aggregate_cue": lambda f: float(f.has_aggregate_cue),
    "max_join_depth": lambda f: _unit(f.max_join_depth / 3.0),
    "n_linked_tables": lambda f: _unit((f.n_linked_tables - 2) / 4.0),
    "n_linked_columns": lambda f: _unit(f.n_linked_columns / 40.0),
    "schema_coverage": lambda f: _unit(f.schema_coverage),
    "ambiguity_score": lambda f: _unit(f.ambiguity_score),
    "link_score_gap": lambda f: 1.0 - _unit(f.link_score_gap / LINK_GAP_SATURATION),
    "link_score_mean": lambda f: 1.0 - _unit(f.link_score_mean / LINK_MEAN_SATURATION),
    "n_tokens": lambda f: _unit(f.n_tokens / 28.0),
    "n_value_candidates": lambda f: _unit(f.n_value_candidates / 5.0),
    "glossary_hits": lambda f: _unit(f.glossary_hits / 3.0),
}


def _unit(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


def heuristic_difficulty(f: DifficultyFeatures) -> float:
    """P(the cheap tier fails), estimated without a trained model.

    This is the cold-start path: a fresh checkout has no ``router_weights.npz``,
    and a system that refuses to route until someone runs a training job is a
    system nobody runs.  It is a plain weighted sum of the normalised terms in
    :data:`HEURISTIC_WEIGHTS` clipped to ``[0, 1]`` — deliberately linear so the
    number stays explainable in the trace, and deliberately *not* tuned against
    the benchmark, so it can never leak test-set information into routing.
    """
    score = sum(w * _NORMALIZERS[k](f) for k, w in HEURISTIC_WEIGHTS.items())
    return _unit(score)


def heuristic_breakdown(f: DifficultyFeatures) -> dict[str, float]:
    """Per-term contributions, largest first — what the UI shows as "why"."""
    parts = {k: round(w * _NORMALIZERS[k](f), 6) for k, w in HEURISTIC_WEIGHTS.items()}
    return dict(sorted(parts.items(), key=lambda kv: -abs(kv[1])))


def top_drivers(f: DifficultyFeatures, k: int = 2) -> list[str]:
    """The ``k`` feature names that contributed most to the heuristic score."""
    return [name for name, value in heuristic_breakdown(f).items() if value > 0.0][:k]


# --------------------------------------------------------------------------- #
# Vectorisation
# --------------------------------------------------------------------------- #


def feature_names() -> list[str]:
    """Canonical feature order — the contract between training and serving."""
    return list(DifficultyFeatures.ORDER)


def to_matrix(features: list[DifficultyFeatures]) -> np.ndarray:
    """Stack features into a ``(n, dim)`` float32 design matrix."""
    dim = DifficultyFeatures.dim()
    if not features:
        return np.zeros((0, dim), dtype=np.float32)
    return np.asarray([f.to_vector() for f in features], dtype=np.float32)


def from_mapping(payload: dict[str, Any]) -> DifficultyFeatures:
    """Rebuild features from a JSONL row, ignoring keys that are not features.

    The training corpus is written by the flywheel and by production traces, so
    it accumulates extra bookkeeping columns; silently dropping them keeps old
    dumps readable after the feature set grows.
    """
    f = DifficultyFeatures()
    for name in DifficultyFeatures.ORDER:
        if name in payload:
            value = payload[name]
            setattr(f, name, int(value) if isinstance(getattr(f, name), int) else float(value))
    return f
