"""FastAPI application.

Two design points worth calling out:

* **The engine is built once, at startup.**  Schema introspection, profiling and
  the retrieval index are expensive and immutable for a given database, so they
  live in the app's lifespan rather than being rebuilt per request.
* **Every response is explainable.**  `/v1/query?explain=true` returns the span
  tree, the schema-linking evidence with per-source scores, the routing reason
  and any governance rewrite — the same information the CLI prints.  An engine
  that cannot explain a refusal is not deployable in a regulated environment.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse

from aegis_sql import __version__
from aegis_sql.api.schemas import (
    EvidenceModel,
    FeedbackRequest,
    HealthResponse,
    LinkRequest,
    LinkResponse,
    PolicyCheckRequest,
    PolicyCheckResponse,
    QueryRequest,
    QueryResponse,
    ViolationModel,
)
from aegis_sql.config import PROJECT_ROOT, Settings, get_settings
from aegis_sql.observability.logging import configure_logging, get_logger
from aegis_sql.observability.metrics import metrics_payload
from aegis_sql.types import Tier

log = get_logger("api")

_ENGINE: Any = None


def get_engine() -> Any:
    if _ENGINE is None:  # pragma: no cover - guarded by lifespan
        raise HTTPException(status_code=503, detail="engine not ready")
    return _ENGINE


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _ENGINE
    from aegis_sql.pipeline import AegisEngine

    settings: Settings = app.state.settings
    configure_logging(settings.log_level, settings.log_json)
    started = time.perf_counter()
    _ENGINE = await asyncio.to_thread(AegisEngine.build, settings)
    log.info("api ready", startup_ms=round((time.perf_counter() - started) * 1000, 1))
    try:
        yield
    finally:
        if _ENGINE is not None:
            _ENGINE.close()
        _ENGINE = None


def create_app(settings: Settings | None = None) -> FastAPI:
    st = settings or get_settings()
    app = FastAPI(
        title="AEGIS-SQL",
        version=__version__,
        description=(
            "한국어 금융·보험 도메인 거버넌스 내장형 Text-to-SQL 엔진.\n\n"
            "`/v1/query` 에 자연어 질문을 보내면 SQL·결과·근거·비용을 함께 돌려준다."
        ),
        docs_url="/docs" if st.server.enable_docs else None,
        lifespan=_lifespan,
    )
    app.state.settings = st
    app.add_middleware(
        CORSMiddleware,
        allow_origins=st.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def _timing(request: Request, call_next):  # noqa: ANN001
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - started) * 1000:.1f}"
        return response

    _register_routes(app)
    return app


def _register_routes(app: FastAPI) -> None:  # noqa: C901 - a route table is naturally long
    # ------------------------------------------------------------------ #
    @app.post("/v1/query", response_model=QueryResponse, tags=["query"])
    async def query(req: QueryRequest, engine=Depends(get_engine)) -> QueryResponse:
        """자연어 질문 → SQL → 실행 결과."""
        tier = Tier(req.tier) if req.tier else None
        bundle = await asyncio.to_thread(
            engine.ask, req.question, req.context, req.allow_clarify, tier
        )
        return QueryResponse.from_bundle(bundle, explain=req.explain, max_rows=req.max_rows)

    # ------------------------------------------------------------------ #
    @app.post("/v1/query/stream", tags=["query"])
    async def query_stream(req: QueryRequest, engine=Depends(get_engine)) -> StreamingResponse:
        """단계별 진행 상황을 SSE로 흘려보낸다 (링킹 → 라우팅 → 생성 → 검증 → 실행)."""
        queue: asyncio.Queue[tuple[str, Any] | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def on_stage(stage: str, payload: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, (stage, payload))

        async def run() -> None:
            tier = Tier(req.tier) if req.tier else None
            try:
                bundle = await asyncio.to_thread(
                    engine.ask, req.question, req.context, req.allow_clarify, tier, on_stage
                )
                await queue.put(
                    ("done", QueryResponse.from_bundle(
                        bundle, explain=req.explain, max_rows=req.max_rows).model_dump())
                )
            except Exception as exc:  # pragma: no cover
                await queue.put(("error", {"message": str(exc)}))
            finally:
                await queue.put(None)

        async def events() -> AsyncIterator[str]:
            task = asyncio.create_task(run())
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    event, payload = item
                    yield f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"
            finally:
                await task

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ------------------------------------------------------------------ #
    @app.post("/v1/link", response_model=LinkResponse, tags=["inspect"])
    async def link(req: LinkRequest, engine=Depends(get_engine)) -> LinkResponse:
        """스키마 링킹 결과와 근거 점수 — 왜 이 컬럼이 선택되었는지."""
        from aegis_sql.schema.card import token_estimate

        nq = engine.c.normalizer.normalize(req.question)
        linked = engine.c.linker.link(nq)
        card = engine.c.card_builder.render(linked, style="mschema")
        return LinkResponse(
            question=req.question,
            intent=nq.intent,
            tokens=nq.tokens[:40],
            entities={k: _jsonable(v) for k, v in nq.entities.items() if v},
            tables=list(linked.tables),
            columns=list(linked.columns),
            glossary=[
                {"term": g.term, "definition": g.definition, "sql_hint": g.sql_hint}
                for g in linked.glossary
            ],
            evidence=[
                EvidenceModel(ref=e.ref, score=round(e.score, 4), source=e.source)
                for e in sorted(linked.evidence, key=lambda e: -e.score)[: req.top]
            ],
            coverage=round(linked.coverage, 4),
            schema_card=card,
            card_tokens=token_estimate(card),
            full_card_tokens=token_estimate(engine.c.card_builder.render(style="mschema")),
        )

    # ------------------------------------------------------------------ #
    @app.post("/v1/policy/check", response_model=PolicyCheckResponse, tags=["governance"])
    async def policy_check(req: PolicyCheckRequest, engine=Depends(get_engine)) -> PolicyCheckResponse:
        """임의의 SQL을 거버넌스 정책에 통과시켜 본다."""
        verdict = engine.c.guard.check(req.sql, req.context)
        return PolicyCheckResponse(
            allowed=verdict.allowed,
            violations=[
                ViolationModel(code=v.code, message=v.message, severity=v.severity, subject=v.subject)
                for v in verdict.violations
            ],
            rewrites=list(verdict.applied_rewrites),
            rewritten_sql=verdict.rewritten_sql,
        )

    # ------------------------------------------------------------------ #
    @app.get("/v1/schema", tags=["inspect"])
    async def schema_info(style: str = "mschema", engine=Depends(get_engine)) -> JSONResponse:
        from aegis_sql.schema.card import token_estimate

        g = engine.c.schema
        card = engine.c.card_builder.render(style=style)
        return JSONResponse(
            {
                "name": g.name,
                "dialect": g.dialect,
                "fingerprint": g.fingerprint(),
                "tables": [
                    {
                        "name": t.name, "comment": t.comment, "rows": t.row_count,
                        "columns": [
                            {
                                "name": c.name, "type": c.dtype, "comment": c.comment,
                                "pk": c.is_primary_key, "code_group": c.code_group,
                                "fk": c.foreign_key.key if c.foreign_key else None,
                                "sensitivity": engine.c.guard.policy.sensitivity(t.name, c.name).value,
                            }
                            for c in t.columns
                        ],
                    }
                    for t in g.tables.values()
                ],
                "foreign_keys": [fk.key for fk in g.foreign_keys],
                "card": card,
                "card_tokens": token_estimate(card),
            }
        )

    # ------------------------------------------------------------------ #
    @app.get("/v1/prompts", tags=["inspect"])
    async def prompts(engine=Depends(get_engine)) -> JSONResponse:
        reg = engine.c.prompt_registry
        return JSONResponse(
            {
                "set": reg.name,
                "manifest": reg.manifest(),
                "prompts": [
                    {"id": r.id, "version": r.version, "hash": r.hash, "role": r.role,
                     "description": r.description, "changelog": r.changelog}
                    for r in reg
                ],
            }
        )

    # ------------------------------------------------------------------ #
    @app.post("/v1/feedback", tags=["flywheel"])
    async def feedback(req: FeedbackRequest) -> JSONResponse:
        """사용자 피드백을 적재한다 — DPO 선호쌍의 원천이 된다."""
        path = PROJECT_ROOT / "data" / "generated" / "feedback.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        record = req.model_dump() | {"ts": time.time()}
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        log.info("feedback recorded", trace_id=req.trace_id, correct=req.correct)
        return JSONResponse({"ok": True, "stored": str(path.relative_to(PROJECT_ROOT))})

    # ------------------------------------------------------------------ #
    @app.get("/v1/health", response_model=HealthResponse, tags=["ops"])
    async def health() -> HealthResponse:
        engine = _ENGINE
        if engine is None:
            raise HTTPException(status_code=503, detail="engine not ready")
        try:
            from aegis_sql.llm.providers import available_providers

            providers = available_providers()
        except Exception:  # pragma: no cover
            providers = {}
        return HealthResponse(
            status="ok",
            version=__version__,
            schema_fingerprint=engine.c.schema.fingerprint(),
            tables=len(engine.c.schema.tables),
            tiers=sorted(t.value for t, g in engine.c.generators.items() if g.available()),
            providers=providers,
            router_loaded=getattr(engine.c.router, "model", None) is not None,
            prompts=engine.c.prompt_registry.manifest(),
        )

    @app.get("/metrics", tags=["ops"], include_in_schema=False)
    async def metrics() -> PlainTextResponse:
        payload, content_type = metrics_payload()
        return PlainTextResponse(payload.decode("utf-8"), media_type=content_type)

    # ------------------------------------------------------------------ #
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def console_page() -> HTMLResponse:
        path = Path(__file__).parent / "console.html"
        if not path.exists():  # pragma: no cover
            return HTMLResponse("<h1>AEGIS-SQL</h1><p>see /docs</p>")
        return HTMLResponse(path.read_text(encoding="utf-8"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


app = None  # populated by uvicorn's factory (`aegis_sql.api.app:create_app`)
