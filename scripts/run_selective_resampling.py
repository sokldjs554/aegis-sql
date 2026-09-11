#!/usr/bin/env python3
"""Run the paired R³-SQL-inspired selective-resampling experiment.

The baseline is one normal AEGIS cascade pass over each answerable KorFin item.
Only rows selected by the frozen trigger receive a fresh, forced ensemble pass;
that observed result replaces the baseline result for the policy score. Raw rows
are appended and fsynced one at a time so a paid run can resume after a runtime
disconnect without inventing or silently dropping outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import zipfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aegis_sql.config import PROJECT_ROOT, Settings
from aegis_sql.eval.harness import BenchItem, load_benchmark
from aegis_sql.eval.metrics import execution_match
from aegis_sql.llm.base import PRICING
from aegis_sql.observability.logging import configure_logging
from aegis_sql.pipeline import AegisEngine
from aegis_sql.research.selective_resampling import (
    ResamplingSignal,
    ResamplingThresholds,
    decide_selective_resampling,
    ids_sha256,
    summarize_live_experiment,
)
from aegis_sql.types import AnswerBundle, AnswerStatus, SQLCandidate, Tier


class StageCapture:
    """Collect the exact stage payloads emitted by one engine query."""

    def __init__(self) -> None:
        self.events: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def __call__(self, stage: str, payload: dict[str, Any]) -> None:
        self.events[stage].append(dict(payload))

    def last(self, stage: str) -> dict[str, Any] | None:
        values = self.events.get(stage, [])
        return values[-1] if values else None


class HostedGenerationFailed(RuntimeError):
    """A provider request failed, so continuing would turn outage into low EX."""


class ObservedCostLimitReached(RuntimeError):
    """The user-approved observed cost ceiling was reached between rows."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
        ).strip()
        return {"commit": commit, "dirty": bool(status)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "", "dirty": True}


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: JSON object expected")
        rows.append(value)
    return rows


def _append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _candidate(candidate: SQLCandidate) -> dict[str, Any]:
    return {
        "sql": candidate.sql,
        "tier": candidate.tier.value,
        "votes": candidate.votes,
        "valid": candidate.valid,
        "error": candidate.error,
        "prompt_version": candidate.prompt_version,
        "raw_output": candidate.raw_output,
    }


def _bundle_audit(bundle: AnswerBundle, capture: StageCapture) -> dict[str, Any]:
    generation = capture.last("generate") or {
        "model": "",
        "requested_samples": 0,
        "completed_samples": 0,
        "candidate_count": len(bundle.candidates),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_usd": 0.0,
        "latency_ms": 0.0,
        "error": "no generation stage observed",
    }
    vote = capture.last("vote")
    candidate_count = len(bundle.candidates)
    if vote is None:
        groups = 1 if candidate_count else 0
        agreement = 1.0 if candidate_count else 0.0
    else:
        groups = int(vote.get("groups", 0))
        agreement = float(vote.get("agreement", 0.0))

    route = bundle.route
    result = bundle.result
    return {
        "status": bundle.status.value,
        "tier": route.tier.value if route else "",
        "route_confidence": float(route.confidence) if route else 1.0,
        "route_difficulty": float(route.difficulty) if route else 0.0,
        "route_reason": route.reason if route else "no route",
        "route_samples": int(route.n_samples) if route else 0,
        "escalated_from": route.escalated_from.value if route and route.escalated_from else None,
        "sql": bundle.sql,
        "executed_sql": bundle.executed_sql,
        "result": {
            "ok": bool(result and result.ok),
            "row_count": result.row_count if result else 0,
            "columns": list(result.columns) if result else [],
            "signature": result.result_signature() if result else None,
            "elapsed_ms": result.elapsed_ms if result else 0.0,
            "error": result.error if result else None,
        },
        "candidate_count": candidate_count,
        "groups": groups,
        "agreement": agreement,
        "vote_observed": vote is not None,
        "vote": vote,
        "generation": generation,
        "generation_attempts": capture.events.get("generate", []),
        "candidates": [_candidate(candidate) for candidate in bundle.candidates],
        "repairs": [
            {
                "attempt": step.attempt,
                "strategy": step.strategy,
                "before_sql": step.before_sql,
                "after_sql": step.after_sql,
                "error": step.error,
                "fixed": step.fixed,
            }
            for step in bundle.repairs
        ],
        "cost_usd": float(bundle.cost_usd),
        "latency_ms": float(bundle.total_latency_ms),
    }


def _assert_provider_calls_succeeded(audit: dict[str, Any], label: str) -> None:
    for generation in audit.get("generation_attempts", []):
        if generation.get("tier") not in {"llm", "ensemble"}:
            continue
        requested = int(generation.get("requested_samples", 0))
        completed = int(generation.get("completed_samples", 0))
        if requested >= 1 and completed == requested:
            continue
        detail = generation.get("error") or "no provider response"
        raise HostedGenerationFailed(
            f"{label}: hosted generation incomplete "
            f"({completed}/{requested} responses): {detail}"
        )


def _correct(bundle: AnswerBundle, gold: Any, gold_sql: str) -> bool:
    return bool(
        bundle.status is AnswerStatus.OK
        and bundle.result is not None
        and execution_match(bundle.result, gold, gold_sql)
    )


def _run_pass(
    engine: AegisEngine,
    item: BenchItem,
    gold: Any,
    *,
    tier: Tier | None,
    provider: str,
    label: str,
) -> tuple[dict[str, Any], bool]:
    capture = StageCapture()
    bundle = engine.ask(
        item.question,
        allow_clarify=True,
        tier=tier,
        on_stage=capture,
        synthesize_answer=False,
    )
    audit = _bundle_audit(bundle, capture)
    if provider != "mock":
        _assert_provider_calls_succeeded(audit, label)
    return audit, _correct(bundle, gold, item.gold_sql or "")


def _make_manifest(
    engine: AegisEngine,
    settings: Settings,
    benchmark_path: Path,
    database_path: Path,
    items: list[BenchItem],
    thresholds: ResamplingThresholds,
    provider_requested: str,
    model_requested: str,
    provider_actual: str,
    model_actual: str,
    max_cost_usd: float | None,
    run_kind: str,
) -> dict[str, Any]:
    router_artifacts = {
        str(path.relative_to(PROJECT_ROOT)): _sha256(path)
        for path in sorted((PROJECT_ROOT / "models" / "router").glob("*"))
        if path.is_file()
    }
    stable = {
        "schema_version": 1,
        "experiment": "r3_selective_resampling",
        "run_kind": run_kind,
        "git": _git_metadata(),
        "benchmark": {
            "path": str(benchmark_path),
            "sha256": _sha256(benchmark_path),
            "items": len(items),
            "selected_ids_sha256": ids_sha256([item.id for item in items]),
        },
        "database": {
            "path": str(database_path),
            "sha256": _sha256(database_path),
            "schema_fingerprint": engine.c.schema.fingerprint(),
        },
        "prompt_manifest": engine.c.prompt_registry.manifest(),
        "router_artifacts_sha256": router_artifacts,
        "provider_requested": provider_requested,
        "provider_actual": provider_actual,
        "model_requested": model_requested,
        "model_actual": model_actual,
        "thresholds": {
            "confidence": thresholds.confidence,
            "agreement": thresholds.agreement,
            "min_groups": thresholds.min_groups,
            "min_candidates": thresholds.min_candidates,
        },
        "protocol": {
            "baseline": "normal AEGIS cascade",
            "resampling": "fresh forced ensemble replaces baseline only on triggered rows",
            "resample_samples": settings.generation.ensemble_samples,
            "answer_synthesis": False,
            "gold_metric": "execution_match",
            "answerable_only": True,
            "few_shot_examples": 0,
            "provider_seed": None,
        },
        "settings": {
            "retrieval": settings.retrieval.model_dump(),
            "generation": settings.generation.model_dump(),
            "router": settings.router.model_dump(),
            "verify": settings.verify.model_dump(),
            "policy": settings.policy.model_dump(),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "packages": {
                name: _version(name)
                for name in (
                    "aegis-sql",
                    "langchain",
                    "langchain-core",
                    "langchain-anthropic",
                    "langchain-openai",
                    "sqlglot",
                )
            },
        },
        "pricing_usd_per_million_tokens": {
            key: {"input": value[0], "output": value[1]}
            for key, value in sorted(PRICING.items())
        },
    }
    created_at = datetime.now(UTC).isoformat()
    return {
        **stable,
        "created_at_utc": created_at,
        "cost_limit_history": [{"at_utc": created_at, "max_cost_usd": max_cost_usd}],
        "configuration_sha256": _canonical_sha256(stable),
    }


def _validate_resume(
    manifest_path: Path,
    raw_path: Path,
    current: dict[str, Any],
    selected_ids: list[str],
    resume: bool,
) -> list[dict[str, Any]]:
    if not manifest_path.exists() and not raw_path.exists():
        if resume:
            print("resume requested, but no prior rows exist; starting a new run", flush=True)
        _atomic_json(manifest_path, current)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.touch(exist_ok=False)
        return []
    if not resume:
        raise FileExistsError(
            "output already exists; pass --resume for the exact same configuration "
            "or choose a new --prefix"
        )
    if not manifest_path.exists() or not raw_path.exists():
        raise FileNotFoundError("resume requires both the manifest and raw JSONL")
    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    if previous.get("configuration_sha256") != current.get("configuration_sha256"):
        raise ValueError("resume configuration differs from the saved manifest")
    history = previous.setdefault("cost_limit_history", [])
    current_limit = (current.get("cost_limit_history") or [{}])[-1]
    history.append(current_limit)
    _atomic_json(manifest_path, previous)
    current.clear()
    current.update(previous)
    rows = _read_rows(raw_path)
    row_ids = [str(row.get("id", "")) for row in rows]
    if row_ids != selected_ids[: len(row_ids)]:
        raise ValueError("saved rows are not an ordered prefix of the selected benchmark")
    return rows


def _write_bundle(path: Path, artifacts: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for artifact in artifacts:
            if artifact.exists():
                archive.write(artifact, arcname=artifact.name)


def _print_summary(summary: dict[str, Any], summary_path: Path, bundle_path: Path) -> None:
    metrics = summary["metrics"]
    base = metrics.get("baseline_accuracy", 0.0)
    policy = metrics.get("policy_accuracy", 0.0)
    cost = metrics["cost_usd"]
    latency = metrics["latency_ms"]
    print("\nR³ selective-resampling result", flush=True)
    print(
        f"trigger: {metrics['triggered']}/{metrics['items']} = {metrics['trigger_rate']:.1%} "
        f"(eligible {metrics['eligible_trigger_rate']:.1%})",
        flush=True,
    )
    print(f"baseline EX: {base:.1%}", flush=True)
    print(f"resampling EX: {policy:.1%}", flush=True)
    print(f"delta EX: {metrics.get('accuracy_delta_pp', 0.0):+.2f} percentage points", flush=True)
    print(
        f"extra cost: ${cost['extra_total']:.6f} "
        f"(${cost['extra_per_trigger']:.6f}/trigger, {cost['increase_ratio']:.1%} vs baseline)",
        flush=True,
    )
    print(
        "latency_ms: "
        f"baseline p50={latency['baseline_p50']:.2f} p95={latency['baseline_p95']:.2f} / "
        f"policy p50={latency['policy_p50']:.2f} p95={latency['policy_p95']:.2f} / "
        f"extra-triggered p50={latency['extra_triggered_p50']:.2f} "
        f"p95={latency['extra_triggered_p95']:.2f}",
        flush=True,
    )
    print(f"portfolio evidence ready: {summary['portfolio_evidence_ready']}", flush=True)
    if summary["evidence_errors"]:
        for error in summary["evidence_errors"]:
            print(f"  - {error}", flush=True)
    print(f"summary: {summary_path}", flush=True)
    print(f"result bundle: {bundle_path}", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run actual paired selective resampling on KorFin-Bench"
    )
    parser.add_argument("--provider", choices=("anthropic", "openai", "mock"), required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--benchmark", default="data/benchmark/korfin_bench.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="smoke rows; omit for all 90")
    parser.add_argument("--ensemble-samples", type=int, default=5)
    parser.add_argument("--confidence-threshold", type=float, default=0.60)
    parser.add_argument("--agreement-threshold", type=float, default=0.60)
    parser.add_argument("--min-groups", type=int, default=2)
    parser.add_argument("--min-candidates", type=int, default=2)
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=None,
        help="stop before the next row once observed cost reaches this value",
    )
    parser.add_argument("--prefix", type=Path, default=Path("reports/r3-selective-resampling"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--expected-items", type=int, default=90)
    parser.add_argument("--log-level", default="WARNING")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.ensemble_samples < 2:
        raise SystemExit("--ensemble-samples must be at least 2")
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise SystemExit("--confidence-threshold must be in [0, 1]")
    if not 0.0 <= args.agreement_threshold <= 1.0:
        raise SystemExit("--agreement-threshold must be in [0, 1]")
    if args.min_groups < 1 or args.min_candidates < 1:
        raise SystemExit("minimum group/candidate thresholds must be positive")
    if args.provider != "mock" and (args.max_cost_usd is None or args.max_cost_usd <= 0.0):
        raise SystemExit("hosted runs require a positive --max-cost-usd")

    configure_logging(args.log_level)
    defaults = {"anthropic": "claude-sonnet-5", "openai": "gpt-4o", "mock": "mock"}
    model = args.model or defaults[args.provider]
    settings = Settings.load(
        generation={
            "provider": args.provider,
            "model": model,
            "ensemble_samples": args.ensemble_samples,
        },
        retrieval={"embedder": "hashing", "vector_store": "numpy", "few_shot_k": 0},
    )
    benchmark_path = _resolve(args.benchmark)
    database_path = _resolve(settings.database.path)
    if not database_path.exists():
        raise SystemExit(f"database not found: {database_path}; run scripts/build_demo_db.py first")

    answerable = [item for item in load_benchmark(benchmark_path) if item.expect == "ok"]
    items = answerable[: args.limit] if args.limit is not None else answerable
    run_kind = "full" if len(items) == args.expected_items and len(items) == len(answerable) else "smoke"
    thresholds = ResamplingThresholds(
        confidence=args.confidence_threshold,
        agreement=args.agreement_threshold,
        min_groups=args.min_groups,
        min_candidates=args.min_candidates,
    )

    engine = AegisEngine.build(settings)
    llm_generator = engine.c.llm_generator
    client = getattr(llm_generator, "client", None)
    provider_actual = str(getattr(client, "name", ""))
    model_actual = str(getattr(client, "model", ""))
    if provider_actual != args.provider:
        engine.close()
        key = "ANTHROPIC_API_KEY" if args.provider == "anthropic" else "OPENAI_API_KEY"
        raise SystemExit(
            f"requested provider {args.provider!r} resolved to {provider_actual!r}; "
            f"install .[llm] and set {key} in the environment (never paste it into a report)"
        )
    if not llm_generator or not llm_generator.available():
        engine.close()
        raise SystemExit(f"provider {args.provider!r} is not available")
    if Tier.ENSEMBLE not in engine.c.generators:
        engine.close()
        raise SystemExit("ensemble generator is unavailable")

    manifest = _make_manifest(
        engine,
        settings,
        benchmark_path,
        database_path,
        items,
        thresholds,
        args.provider,
        model,
        provider_actual,
        model_actual,
        args.max_cost_usd,
        run_kind,
    )
    prefix = args.prefix if args.prefix.is_absolute() else PROJECT_ROOT / args.prefix
    raw_path = prefix.with_name(prefix.name + "-raw.jsonl")
    manifest_path = prefix.with_name(prefix.name + "-manifest.json")
    summary_path = prefix.with_name(prefix.name + "-summary.json")
    bundle_path = prefix.with_name(prefix.name + "-results.zip")

    try:
        rows = _validate_resume(
            manifest_path,
            raw_path,
            manifest,
            [item.id for item in items],
            args.resume,
        )
    except Exception:
        engine.close()
        raise

    completed = {str(row["id"]) for row in rows}
    observed_cost = sum(
        float(row.get("baseline_cost_usd", 0.0))
        + float(row.get("resample_extra_cost_usd", 0.0))
        for row in rows
    )
    run_error: str | None = None
    exit_code = 0
    try:
        for position, item in enumerate(items, 1):
            if item.id in completed:
                continue
            if args.max_cost_usd is not None and observed_cost >= args.max_cost_usd:
                raise ObservedCostLimitReached(
                    f"observed cost ${observed_cost:.6f} reached --max-cost-usd "
                    f"${args.max_cost_usd:.6f}"
                )

            gold = engine.c.executor.execute(item.gold_sql or "")
            baseline, baseline_correct = _run_pass(
                engine,
                item,
                gold,
                tier=None,
                provider=args.provider,
                label=f"{item.id} baseline",
            )
            signal = ResamplingSignal(
                route_confidence=float(baseline["route_confidence"]),
                groups=int(baseline["groups"]),
                agreement=float(baseline["agreement"]),
                candidate_count=int(baseline["candidate_count"]),
            )
            decision = decide_selective_resampling(signal, thresholds)

            resample: dict[str, Any] | None = None
            resampled_correct: bool | None = None
            if decision.trigger:
                resample, resampled_correct = _run_pass(
                    engine,
                    item,
                    gold,
                    tier=Tier.ENSEMBLE,
                    provider=args.provider,
                    label=f"{item.id} resample",
                )

            extra_cost = float(resample["cost_usd"]) if resample else 0.0
            extra_latency = float(resample["latency_ms"]) if resample else 0.0
            policy_correct = resampled_correct if decision.trigger else baseline_correct
            row = {
                "id": item.id,
                "question": item.question,
                "difficulty": item.difficulty,
                "gold_sql": item.gold_sql,
                "gold_result_signature": gold.result_signature(),
                "route_confidence": signal.route_confidence,
                "groups": signal.groups,
                "agreement": signal.agreement,
                "candidate_count": signal.candidate_count,
                "trigger": decision.trigger,
                "trigger_reason": decision.reason,
                "baseline_correct": baseline_correct,
                "resampled_correct": resampled_correct,
                "policy_correct": policy_correct,
                "baseline_cost_usd": float(baseline["cost_usd"]),
                "resample_extra_cost_usd": extra_cost,
                "baseline_latency_ms": float(baseline["latency_ms"]),
                "resample_extra_latency_ms": extra_latency,
                "baseline": baseline,
                "resample": resample,
            }
            _append_row(raw_path, row)
            rows.append(row)
            observed_cost += float(baseline["cost_usd"]) + extra_cost
            outcome = "OK" if baseline_correct else "MISS"
            detail = (
                f" -> R3 {'OK' if resampled_correct else 'MISS'} "
                f"extra=${extra_cost:.6f} {extra_latency:.0f}ms"
                if decision.trigger
                else ""
            )
            print(
                f"[{position:>2}/{len(items)}] {item.id} BASE {outcome} "
                f"tier={baseline['tier'] or '-'} conf={signal.route_confidence:.3f} "
                f"groups={signal.groups} agree={signal.agreement:.3f} "
                f"trigger={'yes' if decision.trigger else 'no'}{detail}",
                flush=True,
            )
    except KeyboardInterrupt:
        run_error = "interrupted by user"
        exit_code = 130
    except (HostedGenerationFailed, ObservedCostLimitReached) as exc:
        run_error = str(exc)
        exit_code = 2
    finally:
        engine.close()

    summary = summarize_live_experiment(
        rows,
        manifest,
        thresholds,
        expected_items=args.expected_items,
    )
    if run_error:
        summary["portfolio_evidence_ready"] = False
        summary["evidence_errors"] = [*summary["evidence_errors"], run_error]
    summary["raw"] = {
        "path": str(raw_path),
        "sha256": _sha256(raw_path) if raw_path.exists() else None,
        "rows": len(rows),
    }
    _atomic_json(summary_path, summary)
    _write_bundle(bundle_path, [manifest_path, raw_path, summary_path])
    _print_summary(summary, summary_path, bundle_path)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
