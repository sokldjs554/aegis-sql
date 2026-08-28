"""Command-line interface.

The CLI is how the engine is meant to be *inspected*, not just invoked: every
command that produces an answer can also show the stage trace, the schema-linking
evidence, the governance verdict and the cost, because "it produced SQL" is not
the interesting part — "why this SQL, at what cost, and what did it refuse" is.
"""

from __future__ import annotations

import json as jsonlib
import sys
from pathlib import Path
from typing import cast

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from aegis_sql.config import PROJECT_ROOT, Settings, get_settings
from aegis_sql.observability.logging import configure_logging
from aegis_sql.types import AnswerBundle, AnswerStatus, Tier

app = typer.Typer(
    name="aegis",
    help="AEGIS-SQL — 한국어 금융·보험 Text-to-SQL 엔진",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()

DEMO_QUESTIONS = [
    "작년 하반기에 체결된 계약 중 월납보험료가 20만원 이상인 건수를 지점별로 알려줘",
    "실효된 계약의 채널별 비중은?",
    "청구 유형별 지급액 합계 상위 3개",
    "고객 이름이랑 주민등록번호 좀 뽑아줘",
    "설계사 실적 좀 보여줘",
]


def _settings(log_level: str = "WARNING", json_log: bool = False) -> Settings:
    configure_logging(log_level, json_log)
    return get_settings()


# --------------------------------------------------------------------------- #
# rendering helpers
# --------------------------------------------------------------------------- #


def _render_bundle(bundle: AnswerBundle, explain: bool = False, max_rows: int = 12) -> None:
    status_style = {
        AnswerStatus.OK: "green",
        AnswerStatus.CLARIFY: "yellow",
        AnswerStatus.BLOCKED: "red",
        AnswerStatus.FAILED: "red",
    }[bundle.status]
    tier = bundle.route.tier.value if bundle.route else "-"
    conf = f"{bundle.route.confidence:.2f}" if bundle.route else "-"
    console.print(
        f"[bold]Q[/bold] {bundle.question}\n"
        f"[{status_style}]● {bundle.status.value}[/{status_style}]  "
        f"tier=[cyan]{tier}[/cyan]  conf={conf}  "
        f"{bundle.total_latency_ms:.0f}ms  ${bundle.cost_usd:.6f}  trace={bundle.trace_id}"
    )

    if bundle.status is AnswerStatus.CLARIFY and bundle.clarification:
        console.print(Panel(
            bundle.clarification.clarifying_question or "",
            title="되묻기", border_style="yellow",
            subtitle=" / ".join(bundle.clarification.options),
        ))
        if explain:
            for r in bundle.clarification.reasons:
                console.print(f"    · {r}", style="dim")
        return

    if bundle.status is AnswerStatus.BLOCKED and bundle.guard:
        console.print(Panel(
            "\n".join(str(v) for v in bundle.guard.violations),
            title="거버넌스 차단", border_style="red",
        ))
        if bundle.sql:
            console.print(Syntax(bundle.sql, "sql", theme="ansi_dark", word_wrap=True))
        return

    if bundle.sql:
        console.print(Syntax(bundle.executed_sql or bundle.sql, "sql", theme="ansi_dark", word_wrap=True))
    if bundle.guard and bundle.guard.applied_rewrites:
        console.print("  재작성: " + ", ".join(bundle.guard.applied_rewrites), style="yellow")

    result = bundle.result
    if result and result.ok and result.rows:
        table = Table(show_header=True, header_style="bold cyan", box=None, pad_edge=False)
        for col in result.columns:
            table.add_column(str(col), overflow="fold")
        for row in result.rows[:max_rows]:
            table.add_row(*["∅" if v is None else _fmt_cell(v) for v in row])
        console.print(table)
        if result.row_count > max_rows:
            console.print(f"  … 총 {result.row_count}행", style="dim")
    if bundle.answer_text:
        console.print(f"[bold green]▸[/bold green] {bundle.answer_text}")

    if explain:
        _render_explain(bundle)


def _fmt_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:,.4f}" if abs(value) < 1000 else f"{value:,.1f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _render_explain(bundle: AnswerBundle) -> None:
    from aegis_sql.observability.trace import render_span

    if bundle.linked:
        console.print("\n[bold]스키마 링킹[/bold]", style="magenta")
        console.print(f"  tables : {', '.join(bundle.linked.tables)}")
        console.print(f"  용어사전: {', '.join(g.term for g in bundle.linked.glossary) or '—'}")
        ev = sorted(bundle.linked.evidence, key=lambda e: -e.score)[:8]
        for e in ev:
            console.print(f"    {e.score:6.3f}  {e.ref:<28} [{e.source}]", style="dim")
    if bundle.route:
        console.print(f"\n[bold]라우팅[/bold]  {bundle.route.reason}", style="magenta")
        console.print(f"  difficulty={bundle.route.difficulty:.3f}  confidence={bundle.route.confidence:.3f}"
                      f"  samples={bundle.route.n_samples}", style="dim")
    if bundle.repairs:
        console.print("\n[bold]자가교정[/bold]", style="magenta")
        for step in bundle.repairs:
            mark = "✓" if step.fixed else "✗"
            console.print(f"  {mark} [{step.strategy}] {step.error[:80]}", style="dim")
    if bundle.trace:
        console.print("\n[bold]트레이스[/bold]", style="magenta")
        console.print(render_span(bundle.trace), style="dim")


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


@app.command()
def ask(
    question: str = typer.Argument(..., help="자연어 질문"),
    explain: bool = typer.Option(False, "--explain", "-e", help="링킹·라우팅·트레이스까지 출력"),
    tier: str | None = typer.Option(None, "--tier", help="티어 강제: template|slm|llm|ensemble"),
    as_json: bool = typer.Option(False, "--json", help="JSON 출력"),
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    """질문 하나를 엔진에 태운다."""
    from aegis_sql.pipeline import AegisEngine

    st = _settings(log_level)
    engine = AegisEngine.build(st)
    bundle = engine.ask(question, tier=Tier(tier) if tier else None)
    if as_json:
        payload = bundle.to_dict()
        payload["trace"] = bundle.trace.to_dict() if bundle.trace else None
        typer.echo(jsonlib.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _render_bundle(bundle, explain=explain)
    engine.close()
    raise typer.Exit(0 if bundle.status in {AnswerStatus.OK, AnswerStatus.BLOCKED,
                                            AnswerStatus.CLARIFY} else 1)


@app.command()
def demo(
    limit: int = typer.Option(5, "--limit", "-n"),
    explain: bool = typer.Option(False, "--explain", "-e"),
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    """대표 질의를 순서대로 실행한다 (거버넌스 차단·되묻기 사례 포함)."""
    from aegis_sql.pipeline import AegisEngine

    st = _settings(log_level)
    engine = AegisEngine.build(st)
    console.rule("[bold]AEGIS-SQL 데모")
    for i, q in enumerate(DEMO_QUESTIONS[:limit], 1):
        console.print(f"\n[bold dim]── {i}/{min(limit, len(DEMO_QUESTIONS))} ──[/bold dim]")
        _render_bundle(engine.ask(q), explain=explain)
    engine.close()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """FastAPI 서버를 띄운다 (웹 콘솔 포함)."""
    import uvicorn

    _settings(log_level)
    uvicorn.run(
        "aegis_sql.api.app:create_app",
        factory=True, host=host, port=port, reload=reload,
        log_level=log_level.lower(),
    )


def _llm_preflight(st: Settings, required: bool) -> None:
    """LLM 티어가 쓰일 평가 전에 키·모델을 실호출 1회로 검증한다.

    검증 없이 시작하면 잘못된 키·소진된 크레딧으로도 문항 수만큼의 실패 호출을
    전부 소진한 뒤 깨끗해 보이는 낮은 EX 리포트가 나온다 — 실제로 관측된 사고
    경로다.  ``required=True`` 는 ``--tier llm/ensemble`` 강제(프로바이더가 없으면
    중단), ``required=False`` 는 캐스케이드 평가(키가 없으면 template 전용이므로
    조용히 통과, 키가 있는데 깨져 있으면 중단)다.
    """
    from aegis_sql.llm.base import Message
    from aegis_sql.llm.mock import MockLLM
    from aegis_sql.llm.providers import get_llm_client

    provider = str(st.generation.provider or "").strip().lower()
    if provider == "template":
        if required:
            console.print(
                "[red]AEGIS_GENERATION__PROVIDER=template 로 고정되어 있어"
                " LLM 티어를 사용할 수 없습니다.[/red] provider 설정을 auto 로 되돌리세요."
            )
            raise typer.Exit(1)
        return
    client = get_llm_client(st, provider)
    if isinstance(client, MockLLM):
        if required:
            console.print(
                "[red]LLM 티어를 강제했지만 사용할 수 있는 LLM 프로바이더가 없습니다.[/red]\n"
                "ANTHROPIC_API_KEY 또는 OPENAI_API_KEY 를 export 했는지,"
                " AEGIS_GENERATION__PROVIDER 가 template 로 고정돼 있지 않은지 확인하세요."
            )
            raise typer.Exit(1)
        return
    model = str(getattr(client, "model", "") or "?")
    try:
        client.complete([Message(role="user", content="ping")], max_tokens=8)
    except Exception as exc:
        console.print(
            f"[red]LLM 사전 점검 실패[/red] — 평가를 시작하지 않습니다.\n"
            f"모델: {model} · 원인: {exc}"
        )
        if not required:
            console.print(
                "LLM 없이 template 티어만 측정하려면 API 키를 unset 하고 다시 실행하세요."
            )
        raise typer.Exit(1) from exc
    console.print(f"LLM 사전 점검 통과 — model={model}")


@app.command(name="eval")
def eval_cmd(
    bench: str | None = typer.Option(None, "--bench", help="벤치마크 jsonl 경로"),
    limit: int | None = typer.Option(None, "--limit", "-n", help="정답 문항 수 제한"),
    difficulty: str | None = typer.Option(None, "--difficulty", help="easy,medium,hard"),
    ablation: bool = typer.Option(False, "--ablation", help="어블레이션 매트릭스 실행"),
    tier: str | None = typer.Option(None, "--tier"),
    report: str | None = typer.Option(None, "--report", help="마크다운 리포트 경로"),
    routing_log: str | None = typer.Option(
        None, "--routing-log", help="라우터 학습용 (features, label) jsonl 출력 경로"
    ),
    show_failures: bool = typer.Option(True, "--failures/--no-failures"),
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    """벤치마크를 돌리고 리포트를 만든다."""
    from aegis_sql.eval.harness import (
        DEFAULT_ABLATIONS,
        EvalHarness,
        Variant,
        write_routing_dataset,
    )
    from aegis_sql.eval.report import failure_details, render_console, write_report

    st = _settings(log_level)
    forced_tier = Tier(tier) if tier else None
    if forced_tier in (Tier.LLM, Tier.ENSEMBLE):
        _llm_preflight(st, required=True)
    elif forced_tier is None:
        # 캐스케이드 평가도 키가 있으면 LLM 티어가 라우팅에 참여한다 — 깨진 키나
        # 소진된 크레딧을 들고 90문항을 도는 것보다 지금 확인하는 편이 낫다.
        _llm_preflight(st, required=False)
    harness = EvalHarness(st, bench_path=bench)
    items = harness.select(
        limit=limit,
        difficulties=[d.strip() for d in difficulty.split(",")] if difficulty else None,
    )
    console.print(f"[bold]평가 시작[/bold] — {len(items)}문항"
                  f" (정답 {sum(1 for i in items if i.expect == 'ok')},"
                  f" 거버넌스 {sum(1 for i in items if i.expect == 'blocked')},"
                  f" 모호성 {sum(1 for i in items if i.expect == 'clarify')})")

    from aegis_sql.eval.harness import BillingExhausted

    try:
        if ablation:
            results = harness.run_ablation(DEFAULT_ABLATIONS, items=items)
        else:
            results = [harness.run_variant(Variant("full", "전체 구성"), items=items,
                                           tier=forced_tier)]
    except BillingExhausted as exc:
        console.print(f"[red]평가 중단[/red] — {exc}")
        raise typer.Exit(1) from exc

    if not results:
        console.print("[red]평가 실패[/red]")
        raise typer.Exit(1)

    console.rule("[bold]결과")
    for r in results:
        console.print(f"[bold cyan]{r.variant}[/bold cyan]  {r.description}")
        console.print(render_console(r))
        console.print()

    if show_failures:
        detail = failure_details(results[0].scores)
        if detail:
            console.rule("[bold]실패 사례")
            console.print(detail, style="dim")

    if routing_log:
        stats = write_routing_dataset(results, routing_log)
        console.print(f"라우터 학습셋: [green]{stats['path']}[/green] "
                      f"({stats['rows']}행, 실패 라벨 {stats['positives']}건)")

    md, js = write_report(results, markdown_path=report or "reports/eval.md")
    console.print(f"\n리포트: [green]{md}[/green]" + (f" · [green]{js}[/green]" if js else ""))


@app.command()
def flywheel(
    n_programs: int = typer.Option(4000, "--n-programs"),
    augment: int = typer.Option(3, "--augment-per-example"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """스키마만으로 학습 데이터를 생성한다 (샘플링→역번역→증강→실행검증)."""
    from aegis_sql.flywheel.build_dataset import build

    st = _settings(log_level)
    stats = build(st, n_programs=n_programs, augment_per_example=augment)
    console.print_json(data=stats)


@app.command()
def profile(log_level: str = typer.Option("INFO", "--log-level")) -> None:
    """컬럼 프로파일 캐시를 만든다."""
    from aegis_sql.schema.introspect import introspect
    from aegis_sql.schema.profile import Profiler

    st = _settings(log_level)
    db = Path(st.database.path)
    if not db.is_absolute():
        db = PROJECT_ROOT / db
    schema = introspect(db)
    cache = PROJECT_ROOT / "data" / "generated" / "profile.json"
    prof = Profiler(db, sample=st.database.profile_sample).profile(schema, cache_path=cache)
    console.print(f"프로파일 {len(prof.columns)}개 컬럼 → [green]{cache}[/green]")


@app.command()
def schema(
    table: str | None = typer.Option(None, "--table", "-t"),
    style: str = typer.Option("mschema", "--style", help="mschema|ddl|compact|slm"),
) -> None:
    """스키마 카드를 출력한다 (프롬프트에 실제로 들어가는 형태)."""
    from aegis_sql.schema.card import SchemaCardBuilder, token_estimate
    from aegis_sql.schema.card import Style as CardStyle
    from aegis_sql.schema.introspect import introspect
    from aegis_sql.schema.profile import Profiler
    from aegis_sql.types import LinkedSchema

    st = _settings()
    db = Path(st.database.path)
    if not db.is_absolute():
        db = PROJECT_ROOT / db
    g = introspect(db)
    prof = Profiler(db).profile(g, cache_path=PROJECT_ROOT / "data" / "generated" / "profile.json")
    builder = SchemaCardBuilder(g, prof)
    linked = LinkedSchema(tables=[table]) if table else None
    card = builder.render(linked, style=cast("CardStyle", style))
    console.print(card)
    console.print(f"\n[dim]≈ {token_estimate(card)} tokens[/dim]")


@app.command()
def link(
    question: str = typer.Argument(...),
    top: int = typer.Option(15, "--top"),
) -> None:
    """스키마 링킹 결과와 근거 점수를 본다."""
    from aegis_sql.pipeline import AegisEngine

    st = _settings()
    engine = AegisEngine.build(st)
    nq = engine.c.normalizer.normalize(question)
    linked = engine.c.linker.link(nq)
    console.print(f"[bold]정규화[/bold] intent={nq.intent} tokens={nq.tokens[:14]}")
    for key, val in nq.entities.items():
        if val:
            console.print(f"  {key}: {val}", style="dim")
    console.print(f"\n[bold]테이블[/bold] {', '.join(linked.tables)}")
    console.print(f"[bold]용어사전[/bold] {', '.join(g.term for g in linked.glossary) or '—'}")
    table = Table("score", "ref", "source", box=None)
    for e in sorted(linked.evidence, key=lambda e: -e.score)[:top]:
        table.add_row(f"{e.score:.3f}", e.ref, e.source)
    console.print(table)
    engine.close()


@app.command()
def policy(
    sql: str = typer.Argument(..., help="검사할 SQL"),
    branch: str | None = typer.Option(None, "--branch", help="세션 컨텍스트: 지점코드"),
    purpose: str | None = typer.Option(None, "--purpose", help="세션 컨텍스트: 조회 목적"),
) -> None:
    """SQL 한 문장을 거버넌스 정책에 통과시켜 본다."""
    from aegis_sql.pipeline import AegisEngine

    st = _settings()
    engine = AegisEngine.build(st)
    ctx = {k: v for k, v in {"branch_cd": branch, "purpose": purpose}.items() if v}
    verdict = engine.c.guard.check(sql, ctx)
    console.print(f"[bold]{'허용' if verdict.allowed else '차단'}[/bold]",
                  style="green" if verdict.allowed else "red")
    for v in verdict.violations:
        console.print(f"  {v}", style="yellow" if v.severity != "block" else "red")
    if verdict.applied_rewrites:
        console.print("  재작성: " + ", ".join(verdict.applied_rewrites))
    if verdict.rewritten_sql:
        console.print(Syntax(verdict.rewritten_sql, "sql", theme="ansi_dark", word_wrap=True))
    engine.close()


prompt_app = typer.Typer(help="프롬프트 레지스트리")
app.add_typer(prompt_app, name="prompt")


@prompt_app.command("list")
def prompt_list(prompt_set: str = typer.Option("default", "--set")) -> None:
    from aegis_sql.prompts.registry import PromptRegistry

    reg = PromptRegistry.load(prompt_set)
    table = Table("id", "version", "hash", "role", "description", box=None)
    for rec in sorted(reg, key=lambda r: r.id):
        table.add_row(rec.id, rec.version, rec.hash, rec.role, rec.description[:52])
    console.print(table)


@prompt_app.command("show")
def prompt_show(prompt_id: str, prompt_set: str = typer.Option("default", "--set")) -> None:
    from aegis_sql.prompts.registry import PromptRegistry

    rec = PromptRegistry.load(prompt_set).get(prompt_id)
    console.print(Panel(rec.template, title=rec.ref, subtitle=rec.description))
    if rec.changelog:
        console.print(Panel(rec.changelog, title="changelog", border_style="dim"))


@app.command()
def version() -> None:
    """버전과 사용 가능한 티어를 보고한다."""
    from aegis_sql import __version__
    from aegis_sql.llm.providers import available_providers

    st = _settings()
    console.print(f"aegis-sql {__version__}")
    console.print(f"  provider  : {st.generation.provider} / {st.generation.model}")
    try:
        console.print(f"  providers : {available_providers()}")
    except Exception as exc:  # pragma: no cover
        console.print(f"  providers : (확인 불가: {exc})")
    for name, mod in (("torch", "torch"), ("tensorflow", "tensorflow"),
                      ("chromadb", "chromadb"), ("langchain", "langchain_core")):
        console.print(f"  {name:<10}: {'사용 가능' if _importable(mod) else '미설치'}",
                      style=None if _importable(mod) else "dim")


def _importable(module: str) -> bool:
    """Import a module purely to see whether it is installed, silently.

    TensorFlow writes oneDNN and absl banners straight to file descriptor 2
    before any Python-level logging config can intervene, so the fd itself has
    to be redirected for the duration of the import.
    """
    import importlib
    import os

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        importlib.import_module(module)
        return True
    except Exception:
        return False
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


def main() -> None:  # pragma: no cover - console entry point
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
