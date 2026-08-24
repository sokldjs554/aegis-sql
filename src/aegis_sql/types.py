"""Core domain model for AEGIS-SQL.

Everything in the engine speaks these types.  They are deliberately plain
dataclasses (not pydantic models) so that the hot path stays allocation-cheap;
pydantic is used only at the FastAPI boundary (`aegis_sql.api.schemas`).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Literal

# --------------------------------------------------------------------------- #
# Schema layer
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ForeignKey:
    """A single-column foreign key edge (composite keys are expanded to one FK per column)."""

    from_table: str
    from_column: str
    to_table: str
    to_column: str

    @property
    def key(self) -> str:
        return f"{self.from_table}.{self.from_column}->{self.to_table}.{self.to_column}"


@dataclass(slots=True)
class ColumnInfo:
    """A physical column enriched with the metadata that schema-linking needs.

    Korean enterprise schemas are the motivating case: `name` is a cryptic
    physical name (``CTRT_STAT_CD``) while `comment` carries the Korean logical
    name (``계약상태코드``).  Both are indexed for retrieval.
    """

    table: str
    name: str
    dtype: str
    nullable: bool = True
    is_primary_key: bool = False
    comment: str | None = None
    #: Set when this column references another table.
    foreign_key: ForeignKey | None = None
    #: Representative values sampled from the table (used for value linking).
    sample_values: list[str] = field(default_factory=list)
    #: NULL-excluded distinct count, ``-1`` when not profiled.
    distinct_count: int = -1
    #: For code columns joined against a code table, the resolved code group.
    code_group: str | None = None
    #: Governance classification, filled in by the policy loader.
    sensitivity: Sensitivity = None  # type: ignore[assignment]

    @property
    def qualified(self) -> str:
        return f"{self.table}.{self.name}"

    @property
    def label(self) -> str:
        """Human-facing label: Korean comment when present, else the physical name."""
        return self.comment or self.name


@dataclass(slots=True)
class TableInfo:
    name: str
    columns: list[ColumnInfo] = field(default_factory=list)
    comment: str | None = None
    row_count: int = -1
    primary_key: list[str] = field(default_factory=list)
    foreign_keys: list[ForeignKey] = field(default_factory=list)

    @property
    def label(self) -> str:
        return self.comment or self.name

    def column(self, name: str) -> ColumnInfo | None:
        lowered = name.lower()
        for col in self.columns:
            if col.name.lower() == lowered:
                return col
        return None

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


class Sensitivity(str, Enum):
    """Column-level governance class (see ``configs/policy/*.yaml``)."""

    PUBLIC = "public"
    #: Aggregatable but never selectable row-by-row (e.g. salary, premium per person).
    INTERNAL = "internal"
    #: Personally identifying — must be masked before it leaves the engine.
    MASKED = "masked"
    #: Never leaves the database under any circumstance (e.g. encrypted RRN).
    FORBIDDEN = "forbidden"


@dataclass(slots=True)
class SchemaGraph:
    """The full database schema plus the FK join graph over it."""

    tables: dict[str, TableInfo] = field(default_factory=dict)
    dialect: str = "sqlite"
    name: str = "default"

    def table(self, name: str) -> TableInfo | None:
        lowered = name.lower()
        for key, tbl in self.tables.items():
            if key.lower() == lowered:
                return tbl
        return None

    def column(self, table: str, column: str) -> ColumnInfo | None:
        tbl = self.table(table)
        return tbl.column(column) if tbl else None

    @property
    def all_columns(self) -> list[ColumnInfo]:
        return [c for t in self.tables.values() for c in t.columns]

    @property
    def foreign_keys(self) -> list[ForeignKey]:
        return [fk for t in self.tables.values() for fk in t.foreign_keys]

    def fingerprint(self) -> str:
        """Stable hash of the structural schema — used as a cache key."""
        parts: list[str] = []
        for tname in sorted(self.tables):
            tbl = self.tables[tname]
            cols = ",".join(f"{c.name}:{c.dtype}" for c in tbl.columns)
            parts.append(f"{tname}({cols})")
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Retrieval layer
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class GlossaryEntry:
    """A business-term → schema-object mapping (사내 용어사전).

    This is what makes ``"유지율"`` resolve to
    ``TB_CTRT.CTRT_STAT_CD`` + a specific predicate, which no amount of raw
    embedding similarity can do on cryptic physical names.
    """

    term: str
    aliases: list[str] = field(default_factory=list)
    definition: str = ""
    tables: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    #: Optional canonical SQL fragment (e.g. ``CTRT_STAT_CD = '01'``).
    sql_hint: str | None = None


@dataclass(slots=True)
class ScoredItem:
    ref: str
    score: float
    source: str = ""  # "dense" | "lexical" | "glossary" | "value" | "fk-expand"

    def __lt__(self, other: ScoredItem) -> bool:  # pragma: no cover - sort helper
        return self.score < other.score


@dataclass(slots=True)
class LinkedSchema:
    """Output of schema linking: the pruned sub-schema handed to the generator."""

    tables: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)  # qualified "TBL.COL"
    glossary: list[GlossaryEntry] = field(default_factory=list)
    join_paths: list[list[ForeignKey]] = field(default_factory=list)
    evidence: list[ScoredItem] = field(default_factory=list)
    #: Recall guardrail — how much of the full schema survived pruning.
    coverage: float = 0.0


@dataclass(slots=True)
class FewShotExample:
    question: str
    sql: str
    difficulty: str = "medium"
    schema_name: str = "default"
    #: Question with values masked out (DAIL-SQL style) for similarity matching.
    masked_question: str = ""
    sql_skeleton: str = ""
    source: str = "curated"


# --------------------------------------------------------------------------- #
# NLU layer
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class NormalizedQuestion:
    raw: str
    normalized: str
    #: Tokens with Korean particles (조사) stripped.
    tokens: list[str] = field(default_factory=list)
    #: e.g. {"금액": [("20만원", 200000)], "기간": [("작년 하반기", ("2025-07-01","2025-12-31"))]}
    entities: dict[str, list[Any]] = field(default_factory=dict)
    #: Values that look like literals to be matched against column contents.
    value_candidates: list[str] = field(default_factory=list)
    intent: str = "select"


@dataclass(slots=True)
class AmbiguityReport:
    is_ambiguous: bool = False
    reasons: list[str] = field(default_factory=list)
    clarifying_question: str | None = None
    options: list[str] = field(default_factory=list)
    score: float = 0.0


# --------------------------------------------------------------------------- #
# Generation layer
# --------------------------------------------------------------------------- #


class Tier(str, Enum):
    """Model tier chosen by the cascade router."""

    TEMPLATE = "template"  # deterministic, zero-cost grammar generator
    SLM = "slm"  # in-house small model (PyTorch)
    LLM = "llm"  # hosted frontier model
    ENSEMBLE = "ensemble"  # multi-sample + self-consistency on the LLM tier


@dataclass(slots=True)
class SQLCandidate:
    sql: str
    tier: Tier = Tier.TEMPLATE
    logprob: float | None = None
    raw_output: str = ""
    prompt_version: str = ""
    #: Filled in after execution-based voting.
    votes: int = 0
    valid: bool | None = None
    error: str | None = None

    def normalized_key(self) -> str:
        return " ".join(self.sql.lower().split())


@dataclass(slots=True)
class GenerationResult:
    candidates: list[SQLCandidate] = field(default_factory=list)
    tier: Tier = Tier.TEMPLATE
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0

    @property
    def best(self) -> SQLCandidate | None:
        return self.candidates[0] if self.candidates else None


# --------------------------------------------------------------------------- #
# Verification & governance layer
# --------------------------------------------------------------------------- #


Severity = Literal["info", "warn", "error", "block"]


@dataclass(slots=True)
class Violation:
    code: str
    message: str
    severity: Severity = "error"
    subject: str | None = None  # column / table / statement fragment

    def __str__(self) -> str:  # pragma: no cover - display helper
        subj = f" [{self.subject}]" if self.subject else ""
        return f"{self.severity.upper()} {self.code}{subj}: {self.message}"


@dataclass(slots=True)
class GuardVerdict:
    allowed: bool = True
    violations: list[Violation] = field(default_factory=list)
    #: SQL after policy rewriting (masking, LIMIT injection, row filters).
    rewritten_sql: str | None = None
    applied_rewrites: list[str] = field(default_factory=list)

    @property
    def blocking(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "block"]


@dataclass(slots=True)
class ExecutionResult:
    ok: bool = False
    columns: list[str] = field(default_factory=list)
    rows: list[tuple[Any, ...]] = field(default_factory=list)
    row_count: int = 0
    elapsed_ms: float = 0.0
    error: str | None = None
    truncated: bool = False

    def result_signature(self) -> str:
        """Order-insensitive hash of the result set — the basis of execution voting."""
        if not self.ok:
            return f"ERR:{(self.error or '')[:80]}"
        norm = sorted("|".join("" if v is None else str(v) for v in row) for row in self.rows)
        payload = f"{len(self.columns)}#" + "\n".join(norm)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(slots=True)
class RepairStep:
    attempt: int
    before_sql: str
    after_sql: str
    error: str
    strategy: str
    fixed: bool = False


# --------------------------------------------------------------------------- #
# Routing layer
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class DifficultyFeatures:
    """Hand-engineered features consumed by the TensorFlow cascade router."""

    n_tokens: int = 0
    n_value_candidates: int = 0
    n_linked_tables: int = 0
    n_linked_columns: int = 0
    max_join_depth: int = 0
    has_aggregate_cue: int = 0
    has_ranking_cue: int = 0
    has_temporal_cue: int = 0
    has_comparison_cue: int = 0
    has_nested_cue: int = 0
    has_ratio_cue: int = 0
    has_set_op_cue: int = 0
    glossary_hits: int = 0
    link_score_mean: float = 0.0
    link_score_gap: float = 0.0
    ambiguity_score: float = 0.0
    schema_coverage: float = 0.0

    ORDER: ClassVar[tuple[str, ...]] = (
        "n_tokens",
        "n_value_candidates",
        "n_linked_tables",
        "n_linked_columns",
        "max_join_depth",
        "has_aggregate_cue",
        "has_ranking_cue",
        "has_temporal_cue",
        "has_comparison_cue",
        "has_nested_cue",
        "has_ratio_cue",
        "has_set_op_cue",
        "glossary_hits",
        "link_score_mean",
        "link_score_gap",
        "ambiguity_score",
        "schema_coverage",
    )

    def to_vector(self) -> list[float]:
        return [float(getattr(self, k)) for k in self.ORDER]

    @classmethod
    def dim(cls) -> int:
        return len(cls.ORDER)


@dataclass(slots=True)
class RouteDecision:
    tier: Tier = Tier.TEMPLATE
    difficulty: float = 0.0  # P(hard) from the router
    confidence: float = 0.0  # calibrated P(the chosen tier succeeds)
    reason: str = ""
    escalated_from: Tier | None = None
    n_samples: int = 1
    features: DifficultyFeatures | None = None


# --------------------------------------------------------------------------- #
# Tracing
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Span:
    name: str
    start_ms: float
    end_ms: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)
    children: list[Span] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return max(0.0, self.end_ms - self.start_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "duration_ms": round(self.duration_ms, 2),
            "attributes": self.attributes,
            "children": [c.to_dict() for c in self.children],
        }


# --------------------------------------------------------------------------- #
# Top-level answer bundle
# --------------------------------------------------------------------------- #


class AnswerStatus(str, Enum):
    OK = "ok"
    CLARIFY = "clarify"  # engine needs a follow-up question from the user
    BLOCKED = "blocked"  # governance refused the query
    FAILED = "failed"  # could not produce executable SQL


@dataclass(slots=True)
class AnswerBundle:
    question: str
    status: AnswerStatus = AnswerStatus.OK
    sql: str | None = None
    executed_sql: str | None = None
    result: ExecutionResult | None = None
    answer_text: str = ""
    route: RouteDecision | None = None
    linked: LinkedSchema | None = None
    guard: GuardVerdict | None = None
    repairs: list[RepairStep] = field(default_factory=list)
    candidates: list[SQLCandidate] = field(default_factory=list)
    clarification: AmbiguityReport | None = None
    trace: Span | None = None
    total_latency_ms: float = 0.0
    cost_usd: float = 0.0
    trace_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "status": self.status.value,
            "sql": self.sql,
            "executed_sql": self.executed_sql,
            "answer_text": self.answer_text,
            "row_count": self.result.row_count if self.result else 0,
            "tier": self.route.tier.value if self.route else None,
            "confidence": round(self.route.confidence, 4) if self.route else None,
            "latency_ms": round(self.total_latency_ms, 2),
            "cost_usd": round(self.cost_usd, 6),
            "trace_id": self.trace_id,
        }


def now_ms() -> float:
    return time.perf_counter() * 1000.0
