"""The orchestrator.

Every other module in this package is deliberately unaware of the others; this
file is the single place where the order of operations lives.  Reading it top to
bottom is the fastest way to understand the system:

    normalize → ambiguity → link → few-shot → route → generate
              → static check → policy guard → execute → repair → vote → answer

Two properties are load-bearing and worth stating up front:

* **No LLM call happens before the router decides.**  Normalisation, linking,
  example selection and difficulty estimation are deterministic, which is what
  makes the cheap tiers viable and the traces reproducible.
* **Nothing reaches the database that has not been through the guard.**  The
  executor is only ever called with `GuardVerdict.rewritten_sql`.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from aegis_sql.config import PROJECT_ROOT, Settings, get_settings
from aegis_sql.generation.base import GenerationContext, Generator
from aegis_sql.observability.logging import get_logger
from aegis_sql.observability.metrics import (
    ESCALATIONS,
    GUARD_BLOCKS,
    LATENCY,
    QUERIES,
    REPAIRS,
    STAGE_LATENCY,
    TOKENS,
)
from aegis_sql.observability.trace import Tracer, trace_context
from aegis_sql.schema.card import SchemaCardBuilder
from aegis_sql.schema.card import Style as CardStyle
from aegis_sql.schema.graph import JoinGraph
from aegis_sql.schema.introspect import introspect
from aegis_sql.schema.profile import Profiler, SchemaProfile
from aegis_sql.types import (
    AnswerBundle,
    AnswerStatus,
    ExecutionResult,
    GenerationResult,
    GuardVerdict,
    LinkedSchema,
    NormalizedQuestion,
    RouteDecision,
    SQLCandidate,
    Tier,
    Violation,
)

log = get_logger("pipeline")

#: Static defects that make a statement *run* and still be wrong.
#:
#: Comparing a ``YYYYMMDD`` column against ``'2025-07-01'`` parses, executes and
#: returns a perfectly formatted zero — which reads like an answer.  There is no
#: reading of the question under which that comparison is intended, so these
#: codes are treated as failures even though the executor reported success.
#: Note that "empty result" is *not* a usable signal here: ``COUNT(*)`` returns
#: one row containing 0, so the defect has to be detected statically.
_SILENT_FAILURE_CODES = frozenset(
    {"DATE_FORMAT_MISMATCH", "DATE_FUNCTION_ON_TEXT", "CODE_LITERAL_MISMATCH"}
)

#: Tiers whose generator consumes a rendered prompt (and therefore few-shots).
_PROMPTED_TIERS = frozenset({Tier.SLM, Tier.LLM, Tier.ENSEMBLE})


def _resolve(path: str | Path) -> Path:
    """Config paths may be relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


@dataclass(slots=True)
class EngineComponents:
    """Everything the engine wires together — exposed so tests can swap parts."""

    settings: Settings
    schema: Any
    profile: SchemaProfile
    join_graph: JoinGraph
    card_builder: SchemaCardBuilder
    normalizer: Any
    ambiguity: Any
    decomposer: Any
    glossary: Any
    linker: Any
    fewshot: Any
    router: Any
    executor: Any
    guard: Any
    static_checker: Any
    intent_guard: Any
    repairer: Any
    generators: dict[Tier, Generator] = field(default_factory=dict)
    llm_generator: Any = None
    prompt_registry: Any = None


class AegisEngine:
    """Question in, :class:`AnswerBundle` out."""

    def __init__(self, components: EngineComponents) -> None:
        self.c = components
        self.settings = components.settings

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #

    @classmethod
    def build(cls, settings: Settings | None = None, **overrides: Any) -> AegisEngine:
        """Wire the whole engine.  Heavy/optional pieces degrade instead of failing."""
        from aegis_sql.nlu.ambiguity import AmbiguityDetector
        from aegis_sql.nlu.decompose import QuestionDecomposer
        from aegis_sql.nlu.korean import KoreanNormalizer
        from aegis_sql.prompts.registry import PromptRegistry
        from aegis_sql.retrieval.embedder import get_embedder
        from aegis_sql.retrieval.fewshot import FewShotSelector
        from aegis_sql.retrieval.glossary import Glossary
        from aegis_sql.retrieval.schema_linker import SchemaLinker
        from aegis_sql.router.cascade import CascadeRouter
        from aegis_sql.verify.ast_guard import PolicyDocument, PolicyGuard
        from aegis_sql.verify.executor import SQLExecutor
        from aegis_sql.verify.intent_guard import RequestIntentGuard
        from aegis_sql.verify.repair import SelfRepairer
        from aegis_sql.verify.static_check import StaticChecker

        st = settings or get_settings()
        if overrides:
            st = Settings.model_validate({**st.model_dump(), **overrides})

        db_path = _resolve(st.database.path)
        schema = introspect(db_path)
        profile = Profiler(db_path, sample=st.database.profile_sample).profile(
            schema, cache_path=_resolve(st.retrieval.persist_dir).parent / "profile.json"
        )
        join_graph = JoinGraph(schema)
        card_builder = SchemaCardBuilder(schema, profile, join_graph)

        glossary_path = db_path.parent / "glossary.yaml"
        glossary = Glossary.load(glossary_path) if glossary_path.exists() else Glossary(entries=[])
        embedder = get_embedder(st)
        linker = SchemaLinker(
            schema=schema, profile=profile, join_graph=join_graph,
            glossary=glossary, embedder=embedder, settings=st,
        )
        linker.build_index()

        fewshot_path = _resolve("data/generated/flywheel/train.jsonl")
        fewshot = None
        if fewshot_path.exists():
            try:
                fewshot = FewShotSelector.from_jsonl(fewshot_path, embedder)
            except Exception as exc:  # pragma: no cover - corrupt or empty corpus
                log.warning("few-shot corpus unusable", path=str(fewshot_path), error=str(exc))

        executor = SQLExecutor(db_path, timeout_s=st.database.timeout_s, max_rows=st.database.max_rows)
        policy = PolicyDocument.load(_resolve(st.policy.path)) if st.policy.enabled else PolicyDocument.permissive()
        guard = PolicyGuard(schema, policy, st)
        intent_guard = RequestIntentGuard(schema, policy=policy, enabled=st.policy.enabled)
        static_checker = StaticChecker(schema, join_graph, profile)
        registry = PromptRegistry.load(st.generation.prompt_set)

        # Columns the policy will refuse or mask.  Handing this to the generator
        # is not a second enforcement point — the guard still runs — it simply
        # stops the cheap tier from spending a round trip on a statement that is
        # certain to be blocked.
        excluded = {
            col.qualified
            for col in schema.all_columns
            if policy.sensitivity(col.table, col.name).value in {"forbidden", "internal"}
        }
        generators, llm_generator = cls._build_generators(
            st, schema, profile, join_graph, glossary, registry, excluded
        )
        repairer = SelfRepairer(
            schema=schema, profile=profile, join_graph=join_graph,
            executor=executor, static_checker=static_checker,
            llm_repair=(llm_generator.repair if llm_generator and llm_generator.available() else None),
            max_attempts=st.verify.max_repair_attempts,
        )
        # The sLLM tier is *built* whenever a checkpoint exists — `--tier slm`
        # must be able to evaluate it — but it only enters automatic routing
        # once `router.enable_slm` says it has earned its place.
        routable = {t for t, g in generators.items() if g.available()}
        if not st.router.enable_slm:
            routable.discard(Tier.SLM)
        router = CascadeRouter(st, router=cls._load_router(st), available_tiers=routable)

        log.info(
            "engine ready",
            tables=len(schema.tables),
            tiers=sorted(t.value for t, g in generators.items() if g.available()),
            fewshot=len(fewshot) if fewshot else 0,
            prompts=len(registry),
        )
        return cls(
            EngineComponents(
                settings=st, schema=schema, profile=profile, join_graph=join_graph,
                card_builder=card_builder, normalizer=KoreanNormalizer(),
                ambiguity=AmbiguityDetector(schema, [e.term for e in glossary.entries]),
                decomposer=QuestionDecomposer(), glossary=glossary, linker=linker,
                fewshot=fewshot, router=router, executor=executor, guard=guard,
                static_checker=static_checker, intent_guard=intent_guard,
                repairer=repairer, generators=generators,
                llm_generator=llm_generator, prompt_registry=registry,
            )
        )

    @staticmethod
    def _build_generators(st, schema, profile, join_graph, glossary, registry, excluded_columns=None):
        from aegis_sql.generation.template_generator import TemplateGenerator

        generators: dict[Tier, Generator] = {
            Tier.TEMPLATE: TemplateGenerator(
                schema, profile, join_graph, glossary, st, excluded_columns=excluded_columns
            )
        }
        llm_generator = None

        provider = st.generation.provider
        if provider != "template":
            try:
                from aegis_sql.generation.llm_generator import LLMGenerator
                from aegis_sql.llm.providers import get_llm_client

                client = get_llm_client(st, prefer=None if provider == "auto" else provider)
                llm_generator = LLMGenerator(client, registry, st)
                if llm_generator.available():
                    generators[Tier.LLM] = llm_generator
                    generators[Tier.ENSEMBLE] = llm_generator
            except Exception as exc:  # pragma: no cover - optional dependency path
                log.warning("LLM tier unavailable", error=str(exc))

        if provider in {"auto", "slm"}:
            try:
                from aegis_sql.generation.slm_generator import SLMGenerator

                slm = SLMGenerator(_resolve(st.generation.slm_checkpoint), st)
                if slm.available():
                    generators[Tier.SLM] = slm
            except Exception as exc:  # pragma: no cover
                log.debug("sLLM tier unavailable", error=str(exc))
        return generators, llm_generator

    @staticmethod
    def _load_router(st):
        """Freshly trained weights win; the checked-in ones are the fallback.

        Shipping a 15 KB numpy router in ``models/router/`` means a clone shows
        real routing confidences on the first ``make demo`` instead of silently
        falling back to the heuristic — and re-training simply shadows it.
        """
        if not st.router.enabled:
            return None
        try:
            from aegis_sql.router.tf_router import load_router
        except Exception as exc:  # pragma: no cover - optional dependency
            log.debug("router module unavailable, using heuristic", error=str(exc))
            return None
        for directory in (_resolve(st.router.model_dir), PROJECT_ROOT / "models" / "router"):
            try:
                router = load_router(directory)
            except Exception as exc:  # pragma: no cover - corrupt weights
                log.debug("router weights unreadable", path=str(directory), error=str(exc))
                continue
            if router is not None:
                log.info("cascade router loaded", path=str(directory))
                return router
        log.info("no trained router found, falling back to the heuristic difficulty score")
        return None

    # ------------------------------------------------------------------ #
    # the query path
    # ------------------------------------------------------------------ #

    def ask(
        self,
        question: str,
        ctx: dict[str, Any] | None = None,
        allow_clarify: bool = True,
        tier: Tier | None = None,
        on_stage: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> AnswerBundle:
        """Run one question end to end.  Never raises for query-level failures."""
        tracer = Tracer("query")
        bundle = AnswerBundle(question=question, trace_id=tracer.trace_id)
        emit = on_stage or (lambda *_: None)

        with trace_context(tracer.trace_id):
            try:
                self._run(question, ctx or {}, allow_clarify, tier, tracer, bundle, emit)
            except Exception as exc:  # pragma: no cover - last-resort safety net
                log.error("pipeline failure", exc_info=True, error=str(exc))
                bundle.status = AnswerStatus.FAILED
                bundle.answer_text = f"질의 처리 중 오류가 발생했습니다: {exc}"

        bundle.trace = tracer.finish()
        bundle.total_latency_ms = bundle.trace.duration_ms
        QUERIES.labels(
            status=bundle.status.value, tier=(bundle.route.tier.value if bundle.route else "none")
        ).inc()
        LATENCY.labels(tier=(bundle.route.tier.value if bundle.route else "none")).observe(
            bundle.total_latency_ms
        )
        return bundle

    def _run(self, question, ctx, allow_clarify, forced_tier, tracer, bundle, emit) -> None:
        c = self.c
        st = self.settings

        # -- 1. normalise ------------------------------------------------ #
        with tracer.span("normalize") as sp:
            nq: NormalizedQuestion = c.normalizer.normalize(question)
            sp.attributes.update(tokens=len(nq.tokens), intent=nq.intent)
            STAGE_LATENCY.labels(stage="normalize").observe(sp.duration_ms)
        emit("normalize", {"intent": nq.intent, "tokens": nq.tokens[:12]})

        # -- 1b. request intent -------------------------------------------- #
        # A destructive request must be named and refused here.  Leaving it to
        # the AST guard would let a read-only generator "answer" a DELETE with a
        # SELECT, which is a silent reinterpretation of the user's ask.
        with tracer.span("intent_guard") as sp:
            intent_violation = c.intent_guard.check(nq)
            sp.attributes["refused"] = intent_violation is not None
        if intent_violation is not None:
            GUARD_BLOCKS.labels(code=intent_violation.code).inc()
            bundle.status = AnswerStatus.BLOCKED
            bundle.guard = GuardVerdict(allowed=False, violations=[intent_violation])
            bundle.answer_text = intent_violation.message
            emit("blocked", {"violations": [str(intent_violation)]})
            return

        # -- 2. link ----------------------------------------------------- #
        with tracer.span("link") as sp:
            linked: LinkedSchema = c.linker.link(nq)
            sp.attributes.update(tables=len(linked.tables), columns=len(linked.columns),
                                 coverage=round(linked.coverage, 3))
            STAGE_LATENCY.labels(stage="link").observe(sp.duration_ms)
        bundle.linked = linked
        emit("link", {"tables": linked.tables, "glossary": [g.term for g in linked.glossary]})

        # -- 3. ambiguity ------------------------------------------------ #
        with tracer.span("ambiguity") as sp:
            report = c.ambiguity.detect(nq, linked)
            sp.attributes.update(ambiguous=report.is_ambiguous, score=round(report.score, 3))
        if allow_clarify and report.is_ambiguous:
            bundle.status = AnswerStatus.CLARIFY
            bundle.clarification = report
            bundle.answer_text = report.clarifying_question or "질문을 조금 더 구체적으로 알려 주세요."
            emit("clarify", {"question": bundle.answer_text, "options": report.options})
            return

        # -- 5. route ---------------------------------------------------- #
        with tracer.span("route") as sp:
            from aegis_sql.router.features import extract_features

            features = extract_features(nq, linked, c.join_graph, report)
            decision: RouteDecision = (
                RouteDecision(tier=forced_tier, reason="tier forced by caller", confidence=1.0,
                              features=features,
                              n_samples=(max(2, int(st.generation.ensemble_samples))
                                         if forced_tier is Tier.ENSEMBLE else 1))
                if forced_tier
                else c.router.decide(features)
            )
            sp.attributes.update(tier=decision.tier.value, difficulty=round(decision.difficulty, 3),
                                 confidence=round(decision.confidence, 3), reason=decision.reason)
        bundle.route = decision
        emit("route", {"tier": decision.tier.value, "confidence": decision.confidence,
                       "reason": decision.reason})

        # -- 5b. few-shot --------------------------------------------------- #
        # Selection runs *after* routing, and only for tiers that read a prompt.
        # Retrieving and re-ranking examples for the template tier — which
        # compiles the question directly and never sees them — cost ~57ms per
        # query on a 10k-example corpus for exactly nothing.
        few_shots: list = []
        if c.fewshot is not None and decision.tier in _PROMPTED_TIERS:
            with tracer.span("fewshot") as sp:
                few_shots = c.fewshot.select(nq, k=st.retrieval.few_shot_k)
                sp.attributes["k"] = len(few_shots)

        # -- 6-11. generate → verify → execute (with escalation) --------- #
        attempted: list[Tier] = []
        while True:
            attempted.append(decision.tier)
            ok = self._attempt(nq, linked, few_shots, decision, ctx, tracer, bundle, emit)
            if ok or len(attempted) >= 3:
                break
            if forced_tier is not None:
                # 강제 티어(--tier)는 단독 티어 측정이 목적이다. 실패했다고 위 티어로
                # 올려버리면 리포트의 티어 라벨이 조용히 오염된다 (llm 측정이 ensemble로 둔갑).
                break
            nxt = c.router.escalate(decision, reason="이전 티어가 실행 가능한 SQL을 만들지 못함")
            if nxt.tier == decision.tier or nxt.tier not in c.generators:
                break
            ESCALATIONS.labels(from_tier=decision.tier.value, to_tier=nxt.tier.value).inc()
            emit("escalate", {"from": decision.tier.value, "to": nxt.tier.value})
            decision = nxt
            bundle.route = decision

        # -- 12. natural-language answer --------------------------------- #
        if bundle.status is AnswerStatus.OK and bundle.result and bundle.result.ok:
            with tracer.span("answer"):
                bundle.answer_text = self._answer_text(bundle)
            emit("answer", {"text": bundle.answer_text})

    # ------------------------------------------------------------------ #

    def _attempt(self, nq, linked, few_shots, decision, ctx, tracer, bundle, emit) -> bool:
        c = self.c
        st = self.settings
        generator = c.generators.get(decision.tier)
        if generator is None or not generator.available():
            # Falling back silently would let an evaluation report label a
            # template-tier number as an sLLM number — the single most damaging
            # kind of measurement bug, because it is invisible in the output.
            fallback = c.generators[Tier.TEMPLATE]
            log.warning(
                "requested tier unavailable, falling back",
                requested=decision.tier.value, used=fallback.tier.value,
            )
            decision.reason = (
                f"{decision.tier.value} 티어를 사용할 수 없어 {fallback.tier.value}로 대체됨"
                f" · {decision.reason}"
            )
            decision.escalated_from = decision.tier
            decision.tier = fallback.tier
            bundle.route = decision
            generator = fallback
            emit("tier_fallback", {"requested": decision.escalated_from.value,
                                   "used": decision.tier.value})

        style = cast(
            "CardStyle", "slm" if decision.tier is Tier.SLM else st.generation.schema_card_style
        )
        card = c.card_builder.render(linked, style=style)
        gctx = GenerationContext(
            question=nq.raw, normalized=nq, linked=linked, schema_card=card,
            few_shots=few_shots, today=c.normalizer.today.strftime("%Y%m%d"), dialect=c.schema.dialect,
            hints=self._hints(linked), n_samples=decision.n_samples,
            temperature=(st.generation.ensemble_temperature if decision.n_samples > 1 else None),
        )

        with tracer.span("generate", tier=decision.tier.value) as sp:
            gen: GenerationResult = generator.generate(gctx)
            sp.attributes.update(candidates=len(gen.candidates), model=gen.model,
                                 prompt_tokens=gen.prompt_tokens, completion_tokens=gen.completion_tokens)
            if gen.error:
                sp.attributes["error"] = gen.error
            STAGE_LATENCY.labels(stage="generate").observe(sp.duration_ms)
        bundle.cost_usd += gen.cost_usd
        TOKENS.labels(kind="prompt").inc(gen.prompt_tokens)
        TOKENS.labels(kind="completion").inc(gen.completion_tokens)
        bundle.candidates = gen.candidates
        if not gen.candidates:
            bundle.status = AnswerStatus.FAILED
            bundle.answer_text = "SQL을 생성하지 못했습니다." + (
                f" (원인: {gen.error})" if gen.error else ""
            )
            return False
        emit("generate", {"sql": gen.candidates[0].sql, "tier": decision.tier.value})

        # self-consistency across candidates (only when we actually sampled several)
        candidate: SQLCandidate = gen.candidates[0]
        if len(gen.candidates) > 1 and st.verify.self_consistency:
            from aegis_sql.verify.selfconsistency import vote

            with tracer.span("vote") as sp:
                winner, stats = vote(gen.candidates, c.executor)
                sp.attributes.update(**{k: round(v, 3) if isinstance(v, float) else v
                                        for k, v in stats.items()})
            if winner is not None:
                candidate = winner
            emit("vote", stats)

        bundle.sql = candidate.sql
        return self._verify_and_execute(candidate.sql, nq, linked, ctx, tracer, bundle, emit)

    def _verify_and_execute(self, sql, nq, linked, ctx, tracer, bundle, emit) -> bool:
        c = self.c

        with tracer.span("static_check") as sp:
            issues: list[Violation] = c.static_checker.check(sql)
            sp.attributes["issues"] = len(issues)
        if issues:
            emit("static_check", {"issues": [str(v) for v in issues]})

        with tracer.span("guard") as sp:
            verdict: GuardVerdict = c.guard.check(sql, ctx)
            sp.attributes.update(allowed=verdict.allowed, violations=len(verdict.violations),
                                 rewrites=len(verdict.applied_rewrites))
        bundle.guard = verdict
        if not verdict.allowed:
            for v in verdict.blocking:
                GUARD_BLOCKS.labels(code=v.code).inc()
            bundle.status = AnswerStatus.BLOCKED
            bundle.answer_text = self._blocked_message(verdict)
            emit("blocked", {"violations": [str(v) for v in verdict.violations]})
            log.warning("query blocked by policy",
                        codes=[v.code for v in verdict.blocking], trace=bundle.trace_id)
            return True  # a policy block is a final answer, not a failure to retry

        executed = verdict.rewritten_sql or sql
        bundle.executed_sql = executed
        with tracer.span("execute") as sp:
            result: ExecutionResult = c.executor.execute(executed)
            sp.attributes.update(ok=result.ok, rows=result.row_count,
                                 elapsed_ms=round(result.elapsed_ms, 2))
            STAGE_LATENCY.labels(stage="execute").observe(sp.duration_ms)
        bundle.result = result

        # A statement can be *wrong without erroring*.  Comparing a YYYYMMDD
        # column against '2025-07-01' parses, runs, and returns zero rows — the
        # single most dangerous failure mode on this schema, because an empty
        # table reads like a real answer.  When the static checker already
        # flagged a mechanically-fixable defect and the result came back empty,
        # treat it as a failure and let the repairer try; the repair is kept only
        # if it actually produces rows.
        silent_defects = [
            v for v in issues
            if v.code in _SILENT_FAILURE_CODES and v.severity in {"error", "block"}
        ]
        if result.ok and silent_defects:
            log.info(
                "statement ran but carries a known-wrong comparison — attempting repair",
                codes=[v.code for v in silent_defects],
            )

        if not result.ok or silent_defects:
            with tracer.span("repair") as sp:
                reason = result.error or "; ".join(v.message for v in silent_defects)
                fixed, steps = c.repairer.repair(
                    executed, reason, self._repair_ctx(nq, linked, executed, result)
                )
                sp.attributes.update(attempts=len(steps), fixed=fixed is not None)
            bundle.repairs = steps
            REPAIRS.labels(fixed=str(fixed is not None).lower()).inc(max(1, len(steps)))
            emit("repair", {"attempts": len(steps), "fixed": fixed is not None,
                            "strategies": [s.strategy for s in steps]})
            if fixed:
                # a repaired statement is untrusted again — re-guard before running it
                reverdict = c.guard.check(fixed, ctx)
                if reverdict.allowed:
                    candidate_sql = reverdict.rewritten_sql or fixed
                    with tracer.span("execute_repaired") as sp:
                        repaired = c.executor.execute(candidate_sql)
                        sp.attributes.update(ok=repaired.ok, rows=repaired.row_count)
                    # Never trade a working answer for a worse one: a repair of a
                    # silently-empty query is accepted only when it finds rows.
                    # Never trade a working answer for a broken one.
                    if repaired.ok:
                        bundle.guard = reverdict
                        bundle.sql = fixed
                        bundle.executed_sql = candidate_sql
                        bundle.result = repaired
                        result = repaired
            if not result.ok:
                bundle.status = AnswerStatus.FAILED
                bundle.answer_text = f"SQL 실행에 실패했습니다: {result.error}"
                return False

        bundle.status = AnswerStatus.OK
        return True

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _hints(self, linked: LinkedSchema) -> dict[str, Any]:
        hints: list[str] = []
        for entry in linked.glossary:
            if entry.sql_hint:
                hints.append(f"{entry.term} → {entry.sql_hint}")
        return {"hints": hints} if hints else {}

    def _repair_ctx(self, nq, linked, sql, result):
        from aegis_sql.verify.repair import RepairContext

        return RepairContext(
            question=nq.raw,
            schema_card=self.c.card_builder.render(linked, style="mschema"),
            linked=linked, sql=sql, error=result.error or "", attempt=0,
        )

    def _blocked_message(self, verdict: GuardVerdict) -> str:
        reasons = "; ".join(v.message for v in verdict.blocking) or "정책 위반"
        return f"데이터 거버넌스 정책에 의해 차단되었습니다. ({reasons})"

    def _answer_text(self, bundle: AnswerBundle) -> str:
        result = bundle.result
        assert result is not None
        if result.row_count == 0:
            return "조건에 해당하는 데이터가 없습니다."

        gen = self.c.llm_generator
        if gen is not None and gen.available():
            text = gen.synthesize_answer(
                bundle.question, bundle.executed_sql or "", result.rows[:20],
                result.columns, result.row_count,
            )
            if text:
                return text
        return _describe_result(result)

    def close(self) -> None:
        with contextlib.suppress(Exception):  # pragma: no cover - best-effort teardown
            self.c.executor.close()


def _describe_result(result: ExecutionResult) -> str:
    """Deterministic Korean summary used when no LLM is available.

    Scalar aggregates read naturally in Korean units (억/만원); wider results get
    a shape description plus the leading row.
    """
    if len(result.rows) == 1 and len(result.columns) == 1:
        value = result.rows[0][0]
        return f"{result.columns[0]}: {_ko_number(value)}"
    head = result.rows[0]
    lead = ", ".join(f"{c}={_ko_number(v)}" for c, v in zip(result.columns, head, strict=False))
    more = f" (총 {result.row_count}행" + (", 상한 도달" if result.truncated else "") + ")"
    return f"{lead}{more}"


def _ko_number(value: Any) -> str:
    if value is None:
        return "없음"
    if isinstance(value, float):
        if abs(value) < 1:
            return f"{value:.2%}" if 0 <= value <= 1 else f"{value:.4f}"
        value = round(value)
    if isinstance(value, int):
        if abs(value) >= 100_000_000:
            return f"약 {value / 100_000_000:.1f}억 ({value:,})"
        if abs(value) >= 10_000:
            return f"약 {value / 10_000:,.0f}만 ({value:,})"
        return f"{value:,}"
    return str(value)
