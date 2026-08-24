"""Pydantic models for the HTTP boundary.

The internal engine speaks dataclasses; this module is the only place where the
domain model is translated into a wire format.  Keeping the translation explicit
(rather than serialising dataclasses directly) means an internal refactor cannot
silently change the public API — and it lets the response carry the *evidence*
(why these columns, which tier, what was rewritten) that makes the system
auditable rather than merely functional.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from aegis_sql.types import AnswerBundle


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, examples=["실효된 계약의 채널별 비중은?"])
    #: Session context consumed by row-level policies (branch scope, purpose).
    context: dict[str, Any] = Field(default_factory=dict)
    #: Force a tier — useful for A/B and for the ablation harness.
    tier: str | None = None
    #: Include the span tree and linking evidence in the response.
    explain: bool = False
    #: When false the engine answers its best guess instead of asking back.
    allow_clarify: bool = True
    max_rows: int = Field(100, ge=1, le=500)


class ViolationModel(BaseModel):
    code: str
    message: str
    severity: str
    subject: str | None = None


class EvidenceModel(BaseModel):
    ref: str
    score: float
    source: str


class RouteModel(BaseModel):
    tier: str
    difficulty: float
    confidence: float
    reason: str
    n_samples: int
    escalated_from: str | None = None


class RepairModel(BaseModel):
    attempt: int
    strategy: str
    error: str
    fixed: bool
    after_sql: str


class ResultModel(BaseModel):
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool
    elapsed_ms: float


class QueryResponse(BaseModel):
    trace_id: str
    question: str
    status: str
    sql: str | None = None
    executed_sql: str | None = None
    answer: str = ""
    result: ResultModel | None = None
    route: RouteModel | None = None
    linked_tables: list[str] = Field(default_factory=list)
    glossary_terms: list[str] = Field(default_factory=list)
    violations: list[ViolationModel] = Field(default_factory=list)
    rewrites: list[str] = Field(default_factory=list)
    repairs: list[RepairModel] = Field(default_factory=list)
    clarification: dict[str, Any] | None = None
    evidence: list[EvidenceModel] = Field(default_factory=list)
    trace: dict[str, Any] | None = None
    latency_ms: float = 0.0
    cost_usd: float = 0.0

    @classmethod
    def from_bundle(cls, b: AnswerBundle, explain: bool = False, max_rows: int = 100) -> QueryResponse:
        result = None
        if b.result is not None and b.result.ok:
            result = ResultModel(
                columns=[str(c) for c in b.result.columns],
                rows=[list(r) for r in b.result.rows[:max_rows]],
                row_count=b.result.row_count,
                truncated=b.result.truncated or b.result.row_count > max_rows,
                elapsed_ms=round(b.result.elapsed_ms, 2),
            )
        return cls(
            trace_id=b.trace_id,
            question=b.question,
            status=b.status.value,
            sql=b.sql,
            executed_sql=b.executed_sql,
            answer=b.answer_text,
            result=result,
            route=(
                RouteModel(
                    tier=b.route.tier.value, difficulty=round(b.route.difficulty, 4),
                    confidence=round(b.route.confidence, 4), reason=b.route.reason,
                    n_samples=b.route.n_samples,
                    escalated_from=b.route.escalated_from.value if b.route.escalated_from else None,
                )
                if b.route
                else None
            ),
            linked_tables=list(b.linked.tables) if b.linked else [],
            glossary_terms=[g.term for g in b.linked.glossary] if b.linked else [],
            violations=[
                ViolationModel(code=v.code, message=v.message, severity=v.severity, subject=v.subject)
                for v in (b.guard.violations if b.guard else [])
            ],
            rewrites=list(b.guard.applied_rewrites) if b.guard else [],
            repairs=[
                RepairModel(attempt=r.attempt, strategy=r.strategy, error=r.error[:300],
                            fixed=r.fixed, after_sql=r.after_sql)
                for r in b.repairs
            ],
            clarification=(
                {
                    "question": b.clarification.clarifying_question,
                    "options": b.clarification.options,
                    "reasons": b.clarification.reasons,
                    "score": round(b.clarification.score, 3),
                }
                if b.clarification
                else None
            ),
            evidence=(
                [
                    EvidenceModel(ref=e.ref, score=round(e.score, 4), source=e.source)
                    for e in sorted(b.linked.evidence, key=lambda e: -e.score)[:25]
                ]
                if (explain and b.linked)
                else []
            ),
            trace=b.trace.to_dict() if (explain and b.trace) else None,
            latency_ms=round(b.total_latency_ms, 2),
            cost_usd=round(b.cost_usd, 6),
        )


class LinkRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top: int = Field(20, ge=1, le=100)


class LinkResponse(BaseModel):
    question: str
    intent: str
    tokens: list[str]
    entities: dict[str, Any]
    tables: list[str]
    columns: list[str]
    glossary: list[dict[str, Any]]
    evidence: list[EvidenceModel]
    coverage: float
    schema_card: str
    card_tokens: int
    full_card_tokens: int


class PolicyCheckRequest(BaseModel):
    sql: str = Field(..., min_length=1, max_length=20000)
    context: dict[str, Any] = Field(default_factory=dict)


class PolicyCheckResponse(BaseModel):
    allowed: bool
    violations: list[ViolationModel]
    rewrites: list[str]
    rewritten_sql: str | None


class FeedbackRequest(BaseModel):
    trace_id: str
    question: str
    sql: str | None = None
    correct: bool
    corrected_sql: str | None = None
    comment: str = ""


class HealthResponse(BaseModel):
    status: str
    version: str
    schema_fingerprint: str
    tables: int
    tiers: list[str]
    providers: dict[str, bool]
    router_loaded: bool
    prompts: dict[str, str]
