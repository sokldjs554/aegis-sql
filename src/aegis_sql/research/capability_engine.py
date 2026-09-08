"""Capability-aware wrapper for the EHRSQL-inspired answerability experiment.

The 30-item research probe showed that a narrow schema/glossary capability gate
can distinguish the curated clear-but-unanswerable questions from paired hard
negatives.  That result is not broad enough to silently change the default
``AegisEngine.ask`` contract, so this adapter makes the abstention path
*executable* while keeping the default engine untouched until broader validation
exists.

Security precedence is preserved: destructive/admin intent is still delegated to
the normal engine and therefore returns its governance ``blocked`` response
before answerability is considered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aegis_sql.nlu.answerability import AnswerabilityDecision, SchemaAnswerabilityDetector
from aegis_sql.pipeline import AegisEngine
from aegis_sql.types import AnswerBundle, Tier


@dataclass(frozen=True, slots=True)
class CapabilityAwareResult:
    """Result envelope that can represent an explicit schema abstention."""

    question: str
    status: str
    answerability: AnswerabilityDecision
    bundle: AnswerBundle | None = None

    @property
    def abstained(self) -> bool:
        return self.status == "unanswerable"

    def to_dict(self) -> dict[str, Any]:
        if self.bundle is not None:
            payload = self.bundle.to_dict()
        else:
            payload = {
                "question": self.question,
                "status": self.status,
                "sql": None,
                "executed_sql": None,
                "answer_text": "",
                "row_count": 0,
                "tier": None,
                "confidence": None,
                "latency_ms": 0.0,
                "cost_usd": 0.0,
                "trace_id": "",
            }
        payload["status"] = self.status
        payload["answerability"] = {
            "answerable": self.answerability.answerable,
            "missing_concepts": list(self.answerability.missing_concepts),
            "reasons": list(self.answerability.reasons),
            "matched_terms": list(self.answerability.matched_terms),
        }
        if self.abstained:
            payload["answer_text"] = " ".join(self.answerability.reasons) or (
                "현재 데이터베이스의 스키마와 용어사전만으로는 이 질문에 답할 수 없습니다."
            )
        return payload


class CapabilityAwareEngine:
    """Run the narrow answerability gate before the normal AEGIS query path.

    This is intentionally an adapter, not a subclass that changes the production
    engine invisibly.  A caller opts into the research behavior explicitly.
    """

    def __init__(
        self,
        engine: AegisEngine,
        detector: SchemaAnswerabilityDetector | None = None,
    ) -> None:
        self.engine = engine
        self.detector = detector or SchemaAnswerabilityDetector(
            engine.c.schema,
            engine.c.glossary.entries,
        )

    @classmethod
    def build(cls, *args: Any, **kwargs: Any) -> CapabilityAwareEngine:
        return cls(AegisEngine.build(*args, **kwargs))

    def ask(
        self,
        question: str,
        ctx: dict[str, Any] | None = None,
        allow_clarify: bool = True,
        tier: Tier | None = None,
        on_stage: Any = None,
    ) -> CapabilityAwareResult:
        """Return ``unanswerable`` only for a known, evidence-missing capability.

        Intent/gov refusal has higher priority than capability abstention.  This
        prevents a destructive request that happens to mention an unavailable
        concept from being mislabeled as a harmless schema miss.
        """

        nq = self.engine.c.normalizer.normalize(question)
        intent_violation = self.engine.c.intent_guard.check(nq)
        if intent_violation is not None:
            bundle = self.engine.ask(
                question,
                ctx=ctx,
                allow_clarify=allow_clarify,
                tier=tier,
                on_stage=on_stage,
            )
            return CapabilityAwareResult(
                question=question,
                status=bundle.status.value,
                answerability=AnswerabilityDecision(answerable=True),
                bundle=bundle,
            )

        decision = self.detector.assess(question)
        if not decision.answerable:
            if on_stage is not None:
                on_stage(
                    "unanswerable",
                    {
                        "missing_concepts": list(decision.missing_concepts),
                        "reasons": list(decision.reasons),
                    },
                )
            return CapabilityAwareResult(
                question=question,
                status="unanswerable",
                answerability=decision,
                bundle=None,
            )

        bundle = self.engine.ask(
            question,
            ctx=ctx,
            allow_clarify=allow_clarify,
            tier=tier,
            on_stage=on_stage,
        )
        return CapabilityAwareResult(
            question=question,
            status=bundle.status.value,
            answerability=decision,
            bundle=bundle,
        )

    def close(self) -> None:
        self.engine.close()
