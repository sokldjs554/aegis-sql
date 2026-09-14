"""Published, machine-readable evidence for the public model-experiment view.

The web console must not contain a second, hand-maintained copy of benchmark
numbers.  This module composes a compact public manifest from the archived
evaluation reports, the recovered Qwen console evidence, and the R3 paired-run
bundle.  The manifest is generated deliberately and copied into the runtime
image; serving it never starts a model or spends API credit.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from aegis_sql.config import PROJECT_ROOT

_REPO_BLOB = "https://github.com/sokldjs554/aegis-sql/blob/main/"
_PUBLISHED = Path("data/research/model_experiment_evidence.json")
_TEMPLATE = Path("reports/eval_template.json")
_CASCADE = Path("reports/eval_llm.json")
_LLM_ONLY = Path("reports/eval_llm_only.json")
_QWEN = Path("data/research/qwen_t4_full_console_evidence.json")
_R3_SUMMARY = Path("data/research/r3_selective_resampling_full/r3-selective-resampling-summary.json")
_R3_RAW = Path("data/research/r3_selective_resampling_full/r3-selective-resampling-raw.jsonl")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _full_run(report: dict[str, Any]) -> dict[str, Any]:
    runs = [run for run in report.get("runs", []) if run.get("variant") == "full"]
    if len(runs) != 1:
        raise ValueError("evaluation report must contain exactly one full run")
    return runs[0]


def _difficulty(run: dict[str, Any]) -> dict[str, float]:
    return {
        level: float(run["per_difficulty"][level]["execution_accuracy"])
        for level in ("easy", "medium", "hard")
    }


def _validate_comparable_runs(*runs: dict[str, Any]) -> tuple[int, int]:
    """Prove that the three headline runs cover the same benchmark snapshot."""

    id_sets = [{item["id"] for item in run["items"]} for run in runs]
    fingerprints = {run["settings"]["schema_fingerprint"] for run in runs}
    if any(len(ids) != len(run["items"]) for ids, run in zip(id_sets, runs, strict=True)):
        raise ValueError("evaluation report contains duplicate item ids")
    if any(ids != id_sets[0] for ids in id_sets[1:]) or len(fingerprints) != 1:
        raise ValueError("tier reports are not comparable benchmark runs")

    answerable_sets = [{item["id"] for item in run["items"] if item["expect"] == "ok"} for run in runs]
    if any(ids != answerable_sets[0] for ids in answerable_sets[1:]):
        raise ValueError("tier reports do not share the same answerable items")
    return len(id_sets[0]), len(answerable_sets[0])


def _tier(key: str, label: str, run: dict[str, Any], model: str) -> dict[str, Any]:
    overall = run["overall"]
    answerable = [item for item in run["items"] if item["expect"] == "ok"]
    total = len(answerable)
    ex = float(overall["execution_accuracy"])
    correct = sum(bool(item["correct"]) for item in answerable)
    if total != int(overall["n"]) or round(correct / total, 4) != round(ex, 4):
        raise ValueError(f"{key} item rows disagree with aggregate EX")
    return {
        "key": key,
        "label": label,
        "model": model,
        "correct": correct,
        "total": total,
        "execution_accuracy": ex,
        "by_difficulty": _difficulty(run),
        "executable_rate": float(overall["executable_rate"]),
        "p50_latency_ms": float(overall["p50_latency_ms"]),
        "p95_latency_ms": float(overall["p95_latency_ms"]),
        "cost_per_query_usd": float(overall["cost_per_query_usd"]),
        "tier_mix": dict(overall["tier_mix"]),
    }


def _source(root: Path, key: str, label: str, relative: Path) -> dict[str, str]:
    path = root / relative
    return {
        "key": key,
        "label": label,
        "path": relative.as_posix(),
        "url": _REPO_BLOB + relative.as_posix(),
        "sha256": _sha256(path),
    }


def build_model_experiment_evidence(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Compose the public manifest from primary, checked-in measurements."""

    template_report = _json(root / _TEMPLATE)
    cascade_report = _json(root / _CASCADE)
    llm_report = _json(root / _LLM_ONLY)
    qwen = _json(root / _QWEN)
    r3 = _json(root / _R3_SUMMARY)

    template = _full_run(template_report)
    cascade = _full_run(cascade_report)
    llm_only = _full_run(llm_report)
    qbase, qadapted = qwen["base"], qwen["adapted"]
    r3_metrics = r3["metrics"]
    total_items, answerable_items = _validate_comparable_runs(template, cascade, llm_only)

    raw_path = root / _R3_RAW
    raw = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line]
    raw_ids = {row["id"] for row in raw}
    raw_checks = {
        "items": len(raw),
        "unique_ids": len(raw_ids),
        "triggered": sum(bool(row["trigger"]) for row in raw),
        "baseline_correct": sum(bool(row["baseline_correct"]) for row in raw),
        "policy_correct": sum(bool(row["policy_correct"]) for row in raw),
    }
    expected_checks = {
        "items": int(r3_metrics["items"]),
        "unique_ids": int(r3_metrics["items"]),
        "triggered": int(r3_metrics["triggered"]),
        "baseline_correct": int(r3_metrics["baseline_correct"]),
        "policy_correct": int(r3_metrics["policy_correct"]),
    }
    if raw_checks != expected_checks:
        raise ValueError("R3 row-level evidence disagrees with its summary")

    sources = [
        _source(root, "template", "Template 평가", _TEMPLATE),
        _source(root, "cascade", "Claude 캐스케이드 평가", _CASCADE),
        _source(root, "llm_only", "Claude LLM 단독 평가", _LLM_ONLY),
        _source(root, "qwen", "Qwen T4 복구 증거", _QWEN),
        _source(root, "r3_summary", "R³ paired-run 요약", _R3_SUMMARY),
        _source(root, "r3_raw", "R³ 90문항 원본", _R3_RAW),
    ]

    return {
        "schema_version": 1,
        "public_demo": {
            "live_tier": "template",
            "live_cost_usd": 0.0,
            "recorded_results_are_live": False,
            "description_ko": (
                "질의 탭은 API 키 없는 template 티어를 실시간 실행합니다. "
                "이 화면의 Claude·Qwen·R³ 수치는 저장된 실측 결과이며 라이브 추론이 아닙니다."
            ),
        },
        "tier_comparison": {
            "benchmark": "KorFin-Bench",
            "total_items": total_items,
            "answerable_items": answerable_items,
            "hosted_model": llm_only["settings"]["model"],
            "runs": [
                _tier("template", "Template", template, "template/v1"),
                _tier("cascade", "캐스케이드", cascade, "template + claude-sonnet-5"),
                _tier("llm_only", "LLM 단독", llm_only, "claude-sonnet-5"),
            ],
            "decision_ko": (
                "LLM 단독은 template이 도달하지 못한 hard 문항을 30% 해결했지만, "
                "캐스케이드는 LLM 단독보다 5문항 낮아 라우터 임계값 재조정이 필요합니다."
            ),
            "cost_scope_ko": (
                "Claude 비용은 보관된 당시 리포트의 SQL 생성 호출 기준입니다. "
                "당시 누락된 보조 답변 합성 비용은 포함하지 않으므로 전체 비용으로 해석하지 않습니다."
            ),
        },
        "qwen": {
            "model": qwen["experiment"]["model"],
            "gpu": qwen["experiment"]["runtime"]["gpu"],
            "items": int(qwen["experiment"]["items"]),
            "dataset_snapshot": qwen["experiment"]["dataset"]["snapshot"],
            "base": {
                "correct": int(qbase["correct"]),
                "total": int(qbase["total"]),
                "execution_accuracy": float(qbase["execution_accuracy"]),
                "by_difficulty": {key: float(value["ex"]) for key, value in qbase["by_difficulty"].items()},
                "latency_ms": dict(qbase["latency_ms"]),
                "peak_cuda_memory_gib": float(qbase["peak_cuda_memory_gib"]),
            },
            "adapted": {
                "correct": int(qadapted["correct"]),
                "total": int(qadapted["total"]),
                "execution_accuracy": float(qadapted["execution_accuracy"]),
                "by_difficulty": {
                    key: float(value["ex"]) for key, value in qadapted["by_difficulty"].items()
                },
                "latency_ms": dict(qadapted["latency_ms"]),
                "peak_cuda_memory_gib": float(qadapted["peak_cuda_memory_gib"]),
            },
            "training": {
                "wall_clock_seconds": float(qwen["training"]["wall_clock_seconds"]),
                "wall_clock_hms": qwen["training"]["wall_clock_hms"],
                "peak_cuda_memory_gib": float(qwen["training"]["peak_cuda_memory_gib"]),
                "trainable_parameters": int(qwen["training"]["trainable_parameters"]),
                "trainable_percent": float(qwen["training"]["trainable_percent"]),
            },
            "paired": {
                "delta_ex_percentage_points": float(qwen["paired_outcomes"]["delta_ex_percentage_points"]),
                "gained": int(qwen["paired_outcomes"]["gained"]),
                "regressed": int(qwen["paired_outcomes"]["regressed"]),
                "unchanged": int(qwen["paired_outcomes"]["unchanged"]),
            },
            "decision": "not_promoted",
            "decision_ko": (
                "순증가는 1문항이었고 p50·p95 지연도 증가해 서비스 기본 티어로 승격하지 않았습니다."
            ),
            "evidence": {
                "status": qwen["evidence_status"],
                "aggregate_measured": True,
                "raw_result_bundle_archived": bool(qwen["raw_result_bundle_archived"]),
                "per_item_statuses_preserved": 180,
                "predicted_sql_archived": False,
                "limitation_ko": (
                    "Colab 초기화 뒤 노트북 콘솔에서 집계와 180개 OK/MISS 판정을 복구했습니다. "
                    "예측 SQL·문항별 지연 원본은 없어 완전한 행 단위 감사 자료로 부르지 않습니다."
                ),
            },
        },
        "selective_resampling": {
            "method": "R³-SQL-inspired selective resampling",
            "model": r3["manifest"]["model_actual"],
            "items": int(r3_metrics["items"]),
            "eligible_items": int(r3_metrics["eligible_items"]),
            "triggered": int(r3_metrics["triggered"]),
            "trigger_rate": float(r3_metrics["trigger_rate"]),
            "baseline": {
                "correct": int(r3_metrics["baseline_correct"]),
                "execution_accuracy": float(r3_metrics["baseline_accuracy"]),
                "cost_usd": float(r3_metrics["cost_usd"]["baseline_total"]),
                "p50_latency_ms": float(r3_metrics["latency_ms"]["baseline_p50"]),
                "p95_latency_ms": float(r3_metrics["latency_ms"]["baseline_p95"]),
            },
            "policy": {
                "correct": int(r3_metrics["policy_correct"]),
                "execution_accuracy": float(r3_metrics["policy_accuracy"]),
                "cost_usd": float(r3_metrics["cost_usd"]["policy_total"]),
                "p50_latency_ms": float(r3_metrics["latency_ms"]["policy_p50"]),
                "p95_latency_ms": float(r3_metrics["latency_ms"]["policy_p95"]),
            },
            "delta_ex_percentage_points": float(r3_metrics["accuracy_delta_pp"]),
            "extra_cost_usd": float(r3_metrics["cost_usd"]["extra_total"]),
            "cost_increase_ratio": float(r3_metrics["cost_usd"]["increase_ratio"]),
            "transitions": dict(r3_metrics["transitions"]),
            "decision": "rejected",
            "decision_ko": (
                "추가 비용과 p95 지연이 늘었지만 gain 0·regression 1이어서 기본 정책 채택을 기각했습니다. "
                "후속 가설은 trigger와 새 후보 acceptance를 분리하는 것입니다."
            ),
            "evidence": {
                "portfolio_evidence_ready": bool(r3["portfolio_evidence_ready"]),
                "evidence_errors": list(r3["evidence_errors"]),
                "raw_rows_archived": len(raw),
                "row_level_bundle_archived": len(raw) == int(r3_metrics["items"]),
            },
        },
        "sources": sources,
    }


@lru_cache(maxsize=1)
def load_published_model_experiment_evidence(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Load the compact manifest shipped with the API runtime."""

    evidence = _json(root / _PUBLISHED)
    if evidence.get("schema_version") != 1:
        raise ValueError("unsupported model experiment evidence schema")
    return evidence
