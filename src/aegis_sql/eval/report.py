"""Report rendering.

The report is the deliverable of an evaluation run, so it carries everything
needed to trust and reproduce the numbers: the ablation matrix, per-difficulty
and per-tag breakdowns, the governance and clarification probes, cost/latency,
and a provenance block with the prompt hashes and schema fingerprint.

Both a Markdown file (for humans and for the README) and a JSON file (for diffing
runs in CI) are written from the same data.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aegis_sql.config import PROJECT_ROOT
from aegis_sql.eval.harness import RunResult
from aegis_sql.eval.metrics import ItemScore


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]], align: str | None = None) -> str:
    sep = align or ("---" + "|:---:" * (len(headers) - 1))
    out = ["| " + " | ".join(headers) + " |", "|" + sep.replace("---", "---") + "|"]
    if align is None:
        out[1] = "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|"
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def render_markdown(results: list[RunResult], title: str = "AEGIS-SQL 평가 리포트") -> str:
    if not results:
        return f"# {title}\n\n(결과 없음)\n"

    base = results[0]
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")
    parts: list[str] = [f"# {title}", ""]
    parts.append(f"생성 시각: {ts}  ·  벤치마크: `{Path(base.settings_digest['benchmark']).name}` "
                 f"({base.settings_digest['benchmark_items']}문항)")
    parts.append("")

    # -- headline -------------------------------------------------------- #
    o = base.overall
    parts += [
        "## 요약",
        "",
        _table(
            ["지표", "값", "설명"],
            [
                ["**실행 정확도 (EX)**", f"**{_pct(o.execution_accuracy)}**", "결과 집합이 정답과 일치"],
                ["Exact Set Match", _pct(o.exact_match), "SQL 문자열 구조 일치 (보조 지표)"],
                ["Skeleton Match", _pct(o.skeleton_match), "질의 구조는 맞고 상수/컬럼만 다름"],
                ["실행 성공률", _pct(o.executable_rate), "오류 없이 실행된 비율"],
                ["VES", f"{o.ves:.3f}", "정확도 × 상대 실행 효율 (BIRD)"],
                ["자가교정 발동률", _pct(o.repair_rate), "실행 실패 후 수리 시도"],
                ["교정 성공률", _pct(o.repair_success), "수리된 질의 중 정답"],
                ["에스컬레이션률", _pct(o.escalation_rate), "상위 티어로 재시도"],
                ["p50 / p95 지연", f"{o.p50_latency_ms:.0f} / {o.p95_latency_ms:.0f} ms", "종단 지연"],
                ["질의당 비용", f"${o.cost_per_query_usd:.6f}", "LLM 토큰 비용"],
            ],
        ),
        "",
        f"티어 분포: `{base.overall.tier_mix}`",
        "",
    ]

    # -- probes ---------------------------------------------------------- #
    g, c = base.governance, base.clarification
    parts += [
        "## 거버넌스 · 모호성 프로브",
        "",
        "정확도만으로는 배포 가능성을 말할 수 없다. 아래 두 지표는 벤치마크의 일부다.",
        "",
        _table(
            ["프로브", "n", "통과율", "실패 항목"],
            [
                ["거버넌스 (요청 거부 / 문장 차단·마스킹)", g["n"], _pct(g["block_rate"]),
                 ", ".join(g["leaked"]) or "—"],
                ["모호성 (반드시 되물음)", c["n"], _pct(c["clarify_rate"]),
                 ", ".join(c["guessed"]) or "—"],
            ],
        ),
        "",
        "거버넌스 프로브는 **각자가 겨냥하는 계층에서** 채점한다. `intent` 프로브(파괴적 요청)는 "
        "종단 거부 여부로, `sql` 프로브는 가드가 해당 문장을 차단·마스킹하는지로 채점한다. "
        f"참고로 종단 거부율은 {_pct(g.get('e2e_refusal_rate', 0.0))} 이며, 이 값은 활성 티어가 "
        "위험한 컬럼을 실제로 생성했는지에 좌우되므로 안전성 수치로 읽으면 안 된다.",
        "",
    ]

    # -- difficulty ------------------------------------------------------ #
    parts += ["## 난이도별", "", _table(
        ["난이도", "n", "EX", "EM", "실행성공", "p50(ms)"],
        [[k, v.n, _pct(v.execution_accuracy), _pct(v.exact_match),
          _pct(v.executable_rate), f"{v.p50_latency_ms:.0f}"]
         for k, v in base.per_difficulty.items()],
    ), ""]

    # -- ablation -------------------------------------------------------- #
    if len(results) > 1:
        parts += ["## 어블레이션", "",
                  "각 행은 **한 가지 구성요소만 제거**하고 동일한 벤치마크를 다시 돌린 결과다.", ""]
        rows = []
        for r in results:
            delta = r.overall.execution_accuracy - base.overall.execution_accuracy
            sign = "—" if r.variant == base.variant else f"{delta * 100:+.1f}%p"
            rows.append([
                f"`{r.variant}`", r.description, _pct(r.overall.execution_accuracy), sign,
                _pct(r.per_difficulty.get("medium", r.overall).execution_accuracy),
                _pct(r.per_difficulty.get("hard", r.overall).execution_accuracy),
                f"{r.overall.p50_latency_ms:.0f}",
            ])
        parts += [_table(["구성", "설명", "EX", "Δ", "medium", "hard", "p50(ms)"], rows), ""]

    # -- tags ------------------------------------------------------------ #
    if base.per_tag:
        parts += ["## 질의 유형별 (태그, n≥3)", "", _table(
            ["태그", "n", "EX"],
            [[f"`{k}`", v.n, _pct(v.execution_accuracy)] for k, v in list(base.per_tag.items())[:18]],
        ), ""]

    # -- failures -------------------------------------------------------- #
    fails = [s for s in base.scores if s.expect == "ok" and not s.correct]
    if fails:
        parts += [f"## 실패 사례 ({len(fails)}건)", "", _table(
            ["id", "난이도", "상태", "원인"],
            [[s.id, s.difficulty, s.status, (s.error or "—")[:90]] for s in fails[:25]],
        ), ""]
        if len(fails) > 25:
            parts.append(f"…외 {len(fails) - 25}건\n")

    # -- repairs --------------------------------------------------------- #
    strategies: dict[str, int] = {}
    for s in base.scores:
        for st_name in s.repair_strategies:
            strategies[st_name] = strategies.get(st_name, 0) + 1
    if strategies:
        parts += ["## 자가교정 전략별 발동 횟수", "", _table(
            ["전략", "횟수"],
            sorted(strategies.items(), key=lambda kv: -kv[1]),
        ), ""]

    # -- provenance ------------------------------------------------------ #
    d = base.settings_digest
    parts += [
        "## 재현 정보",
        "",
        "```json",
        json.dumps(
            {
                "schema_fingerprint": d["schema_fingerprint"],
                "prompt_manifest": d["prompt_manifest"],
                "provider": d["provider"],
                "model": d["model"],
                "available_tiers": d["available_tiers"],
                "embedder": d["embedder"],
                "python": d["python"],
                "wall_s": round(base.wall_s, 1),
            },
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
        "```bash",
        "make setup && make eval          # 이 표를 그대로 재생성",
        "```",
        "",
    ]
    return "\n".join(parts)


def write_report(
    results: list[RunResult],
    markdown_path: str | Path = "reports/eval.md",
    json_path: str | Path | None = None,
    title: str = "AEGIS-SQL 평가 리포트",
) -> tuple[Path, Path | None]:
    md_path = Path(markdown_path)
    if not md_path.is_absolute():
        md_path = PROJECT_ROOT / md_path
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(results, title), encoding="utf-8")

    js_path = None
    if json_path is not False:  # type: ignore[comparison-overlap]
        js_path = Path(json_path) if json_path else md_path.with_suffix(".json")
        if not js_path.is_absolute():
            js_path = PROJECT_ROOT / js_path
        js_path.write_text(
            json.dumps(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "runs": [r.as_dict() for r in results],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return md_path, js_path


def render_console(result: RunResult) -> str:
    """Compact summary for the CLI."""
    o = result.overall
    lines = [
        f"  EX {_pct(o.execution_accuracy)}   EM {_pct(o.exact_match)}   "
        f"실행성공 {_pct(o.executable_rate)}   VES {o.ves:.3f}",
        f"  p50 {o.p50_latency_ms:.0f}ms  p95 {o.p95_latency_ms:.0f}ms  "
        f"비용/질의 ${o.cost_per_query_usd:.6f}  티어 {o.tier_mix}",
        f"  거버넌스 {_pct(result.governance['block_rate'])} ({result.governance['n']}건, "
        f"종단거부 {_pct(result.governance.get('e2e_refusal_rate', 0.0))})   "
        f"모호성 되물음 {_pct(result.clarification['clarify_rate'])} ({result.clarification['n']}건)",
    ]
    for k, v in result.per_difficulty.items():
        lines.append(f"    {k:<7} n={v.n:<3} EX {_pct(v.execution_accuracy)}")
    return "\n".join(lines)


def failure_details(scores: list[ItemScore], limit: int = 10) -> str:
    fails = [s for s in scores if s.expect == "ok" and not s.correct][:limit]
    out = []
    for s in fails:
        out.append(f"  ✗ {s.id} [{s.difficulty}] {s.error[:100]}")
        if s.pred_sql:
            out.append(f"      pred: {s.pred_sql[:140]}")
        if s.gold_sql:
            out.append(f"      gold: {s.gold_sql[:140]}")
    return "\n".join(out)
