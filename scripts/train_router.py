#!/usr/bin/env python3
"""Train the cascade router, calibrate it, and export NumPy weights.

    python scripts/train_router.py --out data/generated/router

Input is a JSONL of ``{"features": {...}, "label": 0|1}`` where **label 1 means
the cheap tier FAILED** — i.e. escalation was necessary.  That is the only
supervision signal worth learning from: it is produced for free by the
evaluation harness and by production traces (a query whose template/sLLM SQL did
not execute, or disagreed with the verified answer, is a positive), so the
router improves as the system runs rather than as someone labels.

If that file does not exist the script synthesises a labelled corpus from
:func:`aegis_sql.router.features.heuristic_difficulty` plus noise, so
``make train-router`` is runnable on a fresh checkout — and says so loudly,
because a router trained on its own prior has learned nothing about the world.

The report at the end is the number that justifies the subsystem: at each
escalation threshold, how much of the frontier-model bill disappears, and how
much answer accuracy that costs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aegis_sql.observability.logging import configure_logging  # noqa: E402
from aegis_sql.router.calibrator import (  # noqa: E402
    CALIBRATOR_FILE,
    TemperatureCalibrator,
    reliability_table,
    roc_auc,
)
from aegis_sql.router.features import from_mapping, heuristic_difficulty, to_matrix  # noqa: E402
from aegis_sql.router.tf_router import DifficultyRouter, NumpyRouter  # noqa: E402
from aegis_sql.types import DifficultyFeatures  # noqa: E402

SYNTHETIC_BANNER = """
============================================================================
  !!  합성 데이터로 학습합니다 (SYNTHETIC TRAINING DATA)  !!
  입력 파일이 없어 heuristic_difficulty + 노이즈로 라벨을 만들었습니다.
  이 라우터는 데모/스모크용이며 실제 난이도를 학습한 것이 아닙니다.
  배포 전에 반드시 `aegis eval` 로그(티어별 성공/실패)로 재학습하세요:
      {path}
============================================================================
"""

#: Cue firing rates and shape parameters roughly matching KorFin-Bench.
_CUE_RATES: dict[str, float] = {
    "has_aggregate_cue": 0.75,
    "has_ranking_cue": 0.22,
    "has_temporal_cue": 0.35,
    "has_comparison_cue": 0.25,
    "has_nested_cue": 0.18,
    "has_ratio_cue": 0.15,
    "has_set_op_cue": 0.08,
}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def load_jsonl(path: Path) -> tuple[list[DifficultyFeatures], np.ndarray]:
    """Read ``{"features": {...}, "label": 0|1}`` rows, skipping malformed lines."""
    features: list[DifficultyFeatures] = []
    labels: list[int] = []
    skipped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            features.append(from_mapping(row["features"]))
            labels.append(int(row["label"]))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            skipped += 1
    if skipped:
        print(f"[warn] {skipped} malformed row(s) skipped in {path}")
    return features, np.asarray(labels, dtype=np.int32)


def synthesize(n: int, seed: int) -> tuple[list[DifficultyFeatures], np.ndarray]:
    """Draw plausible feature vectors and label them with a noisy oracle.

    The oracle is the heuristic itself, pushed through a logistic with a
    temperature of 0.10, so ~15% of rows near the boundary get flipped.  Without
    that noise the network would fit the heuristic exactly and every reported
    metric would be a tautology.
    """
    rng = np.random.default_rng(seed)
    features: list[DifficultyFeatures] = []
    for _ in range(n):
        f = DifficultyFeatures()
        f.n_tokens = int(np.clip(rng.poisson(9) + 2, 2, 40))
        f.n_value_candidates = int(np.clip(rng.poisson(1.2), 0, 6))
        f.n_linked_tables = int(np.clip(1 + rng.binomial(5, 0.35), 1, 6))
        f.n_linked_columns = int(np.clip(round(rng.normal(18, 8)), 2, 45))
        f.max_join_depth = int(np.clip(min(f.n_linked_tables - 1, rng.poisson(1.1)), 0, 4))
        for name, rate in _CUE_RATES.items():
            setattr(f, name, int(rng.random() < rate))
        f.glossary_hits = int(np.clip(rng.poisson(0.6), 0, 4))
        f.link_score_mean = float(np.clip(rng.normal(0.45, 0.15), 0.05, 1.25))
        f.link_score_gap = float(np.clip(rng.exponential(0.12), 0.0, 0.8))
        f.ambiguity_score = float(np.clip(rng.exponential(0.12), 0.0, 0.9))
        f.schema_coverage = float(np.clip(rng.normal(0.35, 0.15), 0.1, 1.0))
        features.append(f)

    difficulty = np.asarray([heuristic_difficulty(f) for f in features])
    p_fail = 1.0 / (1.0 + np.exp(-(difficulty - 0.42) / 0.10))
    labels = (rng.random(n) < p_fail).astype(np.int32)
    return features, labels


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def threshold_sweep(probs: np.ndarray, labels: np.ndarray, thresholds: list[float]) -> list[dict[str, float]]:
    """Cost saving vs accuracy retention at each candidate escalate_threshold.

    The baseline is "always call the frontier model": it costs 1.0 and, by
    assumption, answers everything.  Routing a query to the cheap tier saves its
    entire cost but loses the answer whenever ``label == 1``.  This is the
    trade-off curve the ``router.escalate_threshold`` knob rides.
    """
    n = max(1, probs.size)
    hard = labels == 1
    n_hard = max(1, int(hard.sum()))
    rows: list[dict[str, float]] = []
    for t in thresholds:
        escalated = probs >= t
        cheap = ~escalated
        correct = int((cheap & ~hard).sum()) + int(escalated.sum())
        rows.append(
            {
                "threshold": t,
                "escalated": float(escalated.mean()),
                "cost_saving": float(cheap.mean()),
                "accuracy": correct / n,
                "escalation_recall": float((escalated & hard).sum()) / n_hard,
            }
        )
    return rows


def print_report(
    metrics: dict[str, float],
    holdout_auc: float,
    calibration: dict[str, float],
    sweep: list[dict[str, float]],
    reliability: list[dict[str, object]],
    roundtrip: float,
    out_dir: Path,
) -> None:
    print("\n─ 학습 결과 ────────────────────────────────────────────────────────────")
    print(f"  rows                {int(metrics['n'])}   (escalation rate {metrics['pos_rate']:.1%})")
    print(f"  epochs ran          {int(metrics['epochs_ran'])}")
    print(f"  AUC  train / val    {metrics['auc']:.4f} / {metrics['val_auc']:.4f}")
    print(f"  ACC  train / val    {metrics['acc']:.4f} / {metrics['val_acc']:.4f}")
    print(f"  val_loss            {metrics['val_loss']:.4f}")
    print(f"  AUC  holdout        {holdout_auc:.4f}")

    print("\n─ 보정 (Platt/temperature scaling, holdout) ────────────────────────────")
    print(f"  T / b               {calibration['temperature']:.4f} / {calibration['bias']:+.4f}")
    print(f"  NLL  before/after   {calibration['nll_before']:.4f} → {calibration['nll_after']:.4f}")
    print(f"  ECE  before/after   {calibration['ece_before']:.4f} → {calibration['ece_after']:.4f}")
    if calibration.get("identity_fallback"):
        print("  (NLL 최적 T가 ECE를 악화시켜 T=1 로 되돌렸습니다)")

    print("\n─ 신뢰도 표 (보정 후) ──────────────────────────────────────────────────")
    print(f"  {'구간':<14}{'건수':>7}{'예측':>9}{'실제':>9}{'gap':>9}")
    for row in reliability:
        span = f"{row['lower']:.1f}–{row['upper']:.1f}"
        print(f"  {span:<14}{row['count']:>7}{row['mean_prob']:>9.3f}"
              f"{row['empirical']:>9.3f}{row['gap']:>9.3f}")

    print("\n─ 임계값 스윕 (baseline = 모든 질의를 LLM으로) ─────────────────────────")
    print(f"  {'threshold':>10}{'escalated':>12}{'cost saving':>14}{'accuracy':>11}{'hard recall':>13}")
    for row in sweep:
        print(f"  {row['threshold']:>10.2f}{row['escalated']:>11.1%}{row['cost_saving']:>14.1%}"
              f"{row['accuracy']:>11.1%}{row['escalation_recall']:>13.1%}")

    print(f"\n  keras ↔ numpy 최대 오차   {roundtrip:.2e}   (허용 1e-5)")
    print(f"  → {out_dir}\n")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the AEGIS-SQL cascade router")
    ap.add_argument("--data", default="data/generated/router/routing_train.jsonl",
                    help="JSONL of {'features': {...}, 'label': 0|1}")
    ap.add_argument("--out", default="data/generated/router", help="model output directory")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--val-split", type=float, default=0.2, help="validation split inside training")
    ap.add_argument("--holdout", type=float, default=0.2, help="split held out for calibration")
    ap.add_argument("--synthetic-n", type=int, default=3000, help="rows to synthesise when --data is missing")
    ap.add_argument("--bins", type=int, default=10, help="reliability-table bins")
    ap.add_argument("--seed", type=int, default=20260824)
    args = ap.parse_args()

    configure_logging("INFO")
    data_path = Path(args.data) if Path(args.data).is_absolute() else ROOT / args.data
    out_dir = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out

    synthetic = not data_path.exists()
    if synthetic:
        print(SYNTHETIC_BANNER.format(path=data_path))
        features, labels = synthesize(args.synthetic_n, args.seed)
    else:
        features, labels = load_jsonl(data_path)
        print(f"[data] {len(features)} rows from {data_path}")

    if len(features) < 50:
        print(f"[error] 학습에 최소 50행이 필요합니다 (현재 {len(features)}행).", file=sys.stderr)
        return 2
    if len(np.unique(labels)) < 2:
        print("[error] 라벨이 한 종류뿐입니다 — 성공/실패가 모두 있어야 학습됩니다.", file=sys.stderr)
        return 2

    X = to_matrix(features)
    y = labels.astype(np.float32)

    # Held-out split for calibration and the sweep: never seen by fit() or by
    # early stopping, so the ECE and the cost/accuracy curve are honest.
    order = np.random.default_rng(args.seed).permutation(X.shape[0])
    X, y = X[order], y[order]
    cut = int(round(X.shape[0] * (1.0 - args.holdout)))
    X_fit, y_fit, X_hold, y_hold = X[:cut], y[:cut], X[cut:], y[cut:]

    router = DifficultyRouter(out_dir)
    metrics = router.train(X_fit, y_fit, epochs=args.epochs, val_split=args.val_split, seed=args.seed)

    hold_probs = router.model.predict(X_hold, verbose=0).ravel().astype(np.float64)
    holdout_auc = roc_auc(y_hold, hold_probs)

    calibrator = TemperatureCalibrator()
    calibration = calibrator.fit(hold_probs, y_hold)
    calibrated = calibrator.transform(hold_probs)

    router.export_numpy()
    router.save()
    calibrator.save(out_dir / CALIBRATOR_FILE)

    # Persist the holdout evaluation next to the weights.  A number that lives
    # only in a training run's stdout cannot back a README claim; anything the
    # report cites has to be re-checkable from the shipped artefacts alone.
    meta_path = out_dir / "router_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["holdout"] = {
        "n": int(y_hold.shape[0]),
        "auc": round(float(holdout_auc), 6),
        "ece_before": round(float(calibration["ece_before"]), 6),
        "ece_after": round(float(calibration["ece_after"]), 6),
        "temperature": round(float(calibration["temperature"]), 6),
        "sweep": threshold_sweep(calibrated, y_hold, [0.30, 0.40, 0.50, 0.60, 0.70]),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    served = NumpyRouter.load(out_dir)
    if served is None:
        print("[error] export 후 numpy 가중치를 다시 읽지 못했습니다.", file=sys.stderr)
        return 1
    roundtrip = float(np.max(np.abs(served.predict_batch(X_hold) - hold_probs)))

    print_report(
        metrics=metrics,
        holdout_auc=holdout_auc,
        calibration=calibration,
        sweep=threshold_sweep(calibrated, y_hold, [0.30, 0.40, 0.50, 0.60, 0.70]),
        reliability=reliability_table(calibrated, y_hold, n_bins=args.bins),
        roundtrip=roundtrip,
        out_dir=out_dir,
    )
    if synthetic:
        print(SYNTHETIC_BANNER.format(path=data_path))
    if roundtrip >= 1e-5:
        print(f"[error] keras/numpy 불일치 {roundtrip:.2e} — export 경로를 확인하세요.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
