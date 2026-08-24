"""Confidence calibration for the router — making the number mean something.

The router emits a sigmoid.  A sigmoid is not a probability, and three separate
things in this system break when it is treated as one:

1. **The training objective was deliberately distorted.**  Escalations are the
   minority class, so :meth:`~aegis_sql.router.tf_router.DifficultyRouter.train`
   applies ``class_weight`` to stop the model collapsing to "everything is
   easy".  That re-weighting shifts the implied prior: the raw output is now
   ``P(hard)`` under a re-balanced distribution that no production traffic ever
   has.  Reading it as a frequency over real queries is a category error.
2. **The thresholds are business knobs in probability units.**
   ``router.escalate_threshold = 0.55`` is a statement about how much failure
   risk is worth ~$0.009 of frontier-model spend.  If the network's 0.55 really
   corresponds to a 78% failure rate — the textbook over-confidence of modern
   networks (Guo et al., ICML 2017) — then the knob silently does something
   entirely different from what its name and its config comment promise, and
   tuning it becomes guesswork.
3. **Cost accounting stops working.**  The cascade compares an expected cost of
   escalating against an expected cost of being wrong.  ``E[cost]`` is a
   probability times a price; with an uncalibrated probability the comparison is
   arithmetic on a number with no units.

Platt scaling on the logits is the minimal fix: ``p' = sigmoid(logit(p)/T + b)``,
fitted on a *held-out* split by minimising NLL.  Both parameters earn their
place, and they fix different faults:

* ``T`` is the textbook temperature (Guo et al.).  It corrects **sharpness** —
  a model that is right but far too certain about it.
* ``b`` corrects the **prior shift** that reason (1) above bakes in.  Class
  weighting moves the intercept, and no amount of temperature can move an
  intercept: on this system's own training run a temperature-only fit takes ECE
  from 0.176 only to 0.167, while adding the bias takes it below 0.03.  Fixing
  sharpness while leaving a systematic +0.25 offset in place would be calibration
  theatre.  Pin ``b = 0`` and this is exactly temperature scaling.

Neither parameter can change the ranking of two queries — the map is strictly
monotone in the logit — so AUC and every order-dependent routing decision are
untouched; only the claim the number makes about the world changes.  Two degrees
of freedom is also why this does not overfit the few hundred held-out rows that
are all a young system has.

One deviation from the textbook: after fitting, the result is compared against
the identity transform on the fit data and *rejected* if it does not improve ECE.
Minimising NLL usually improves ECE but is not guaranteed to, and shipping a
calibrator that is worse than doing nothing is not a trade this system makes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from aegis_sql.observability.logging import get_logger

log = get_logger("router.calibrator")

CALIBRATOR_FILE = "calibrator.json"

#: Probabilities are clipped before the logit so that a hard 0.0/1.0 from an
#: exported model cannot produce an infinite logit.
_EPS = 1e-6
#: Search bounds for the temperature.  T<1 sharpens, T>1 flattens.
_T_MIN, _T_MAX = 0.05, 20.0
#: Search bounds for the logit bias.  ±6 logits is ±(0.25%…99.75%) of prior shift.
_B_MIN, _B_MAX = -6.0, 6.0
_GRID = 41
#: Coordinate-descent rounds over (T, b).  The surface is smooth and nearly
#: separable, so this converges in two; four is free insurance.
_ROUNDS = 4


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based ROC-AUC with proper tie handling; ``nan`` if one class is absent.

    Implemented via the Mann–Whitney U identity so there is no dependency on
    scikit-learn — this number is printed by the training report and by the
    evaluation harness, both of which must run in a bare environment.
    """
    y = np.asarray(labels, dtype=np.float64).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    pos = float((y > 0.5).sum())
    neg = float(y.size - pos)
    if pos == 0.0 or neg == 0.0:
        return float("nan")
    ranks = _average_ranks(s)
    return float((ranks[y > 0.5].sum() - pos * (pos + 1.0) / 2.0) / (pos * neg))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks, ties sharing their mean rank."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    for i in range(1, values.size + 1):
        if i == values.size or sorted_values[i] != sorted_values[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return ranks


def _bin_stats(probs: np.ndarray, labels: np.ndarray, n_bins: int) -> list[tuple[int, int, float, float]]:
    """``(bin_index, count, mean_prob, empirical_rate)`` for non-empty bins."""
    p = np.clip(np.asarray(probs, dtype=np.float64).ravel(), 0.0, 1.0)
    y = np.asarray(labels, dtype=np.float64).ravel()
    # Right-closed bins so that p == 1.0 lands in the last bin, not a phantom one.
    idx = np.clip(np.ceil(p * n_bins).astype(int) - 1, 0, n_bins - 1)
    out: list[tuple[int, int, float, float]] = []
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count:
            out.append((b, count, float(p[mask].mean()), float(y[mask].mean())))
    return out


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Count-weighted mean gap between predicted probability and observed rate.

    This is the *positive-class* form of ECE rather than the top-label form: the
    router's output is ``P(hard)``, and the question the escalation threshold
    asks is "of the queries scored 0.7, did 70% actually need escalation?".
    """
    p = np.asarray(probs, dtype=np.float64).ravel()
    if p.size == 0:
        return 0.0
    stats = _bin_stats(p, labels, n_bins)
    return float(sum(count * abs(empirical - mean_p) for _, count, mean_p, empirical in stats) / p.size)


def reliability_table(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> list[dict[str, Any]]:
    """Per-bin reliability rows for the evaluation report (empty bins omitted)."""
    total = max(1, np.asarray(probs).size)
    rows: list[dict[str, Any]] = []
    for b, count, mean_p, empirical in _bin_stats(probs, labels, n_bins):
        rows.append(
            {
                "bin": b,
                "lower": round(b / n_bins, 4),
                "upper": round((b + 1) / n_bins, 4),
                "count": count,
                "share": round(count / total, 4),
                "mean_prob": round(mean_p, 4),
                "empirical": round(empirical, 4),
                "gap": round(empirical - mean_p, 4),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Temperature scaling
# --------------------------------------------------------------------------- #


def _logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(p, dtype=np.float64).ravel(), _EPS, 1.0 - _EPS)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def _nll(logits: np.ndarray, labels: np.ndarray, temperature: float, bias: float = 0.0) -> float:
    """Mean binary cross-entropy of ``sigmoid(logits / T + b)``, computed stably."""
    z = logits / temperature + bias
    return float(np.mean(np.maximum(z, 0.0) - z * labels + np.log1p(np.exp(-np.abs(z)))))


def _golden_section(fn: Any, low: float, high: float, tol: float = 1e-6, max_iter: int = 200) -> float:
    """Minimise a unimodal ``fn`` on ``[low, high]`` without derivatives."""
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = high - inv_phi * (high - low)
    d = low + inv_phi * (high - low)
    fc, fd = fn(c), fn(d)
    for _ in range(max_iter):
        if high - low < tol:
            break
        if fc < fd:
            high, d, fd = d, c, fc
            c = high - inv_phi * (high - low)
            fc = fn(c)
        else:
            low, c, fc = c, d, fd
            d = low + inv_phi * (high - low)
            fd = fn(d)
    return (low + high) / 2.0


class TemperatureCalibrator:
    """Platt calibrator on logits: ``p' = sigmoid(logit(p) / T + b)``.

    ``T > 1`` flattens an over-confident model and ``T < 1`` sharpens an
    under-confident one; ``b`` shifts the implied base rate, which is what
    corrects the prior distortion introduced by training with ``class_weight``.
    ``(T, b) == (1, 0)`` is the identity, and ``b == 0`` is textbook temperature
    scaling.  The map is strictly increasing in the logit for any ``T > 0``, so
    calibration never changes which of two queries is considered harder — only
    what the numbers claim about the world.
    """

    __slots__ = ("temperature", "bias", "report")

    def __init__(self, temperature: float = 1.0, bias: float = 0.0) -> None:
        self.temperature = float(temperature)
        self.bias = float(bias)
        self.report: dict[str, float] = {}

    # -- fitting ----------------------------------------------------------- #

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        """Fit ``(T, b)`` on held-out predictions and return a before/after report.

        The search is coordinate descent over two 1-D problems: a coarse
        log-spaced grid to bracket ``T``, then golden-section refinement, then
        the same for ``b``, repeated.  NLL is convex in the logit-affine
        parameters and the two coordinates are nearly orthogonal here, so this
        lands on the same optimum an LBFGS run would — without adding an
        optimiser dependency to a package whose serving path is NumPy only.
        """
        p = np.asarray(probs, dtype=np.float64).ravel()
        y = np.asarray(labels, dtype=np.float64).ravel()
        if p.size != y.size:
            raise ValueError(f"probs {p.shape} and labels {y.shape} do not line up")
        if p.size < 2 or len(np.unique(y)) < 2:
            self.temperature, self.bias = 1.0, 0.0
            self.report = {"temperature": 1.0, "bias": 0.0, "n": float(p.size), "degenerate": 1.0}
            log.warning("calibration skipped — need both classes", n=int(p.size))
            return dict(self.report)

        logits = _logit(p)
        temperature, bias = self._search(logits, y)

        ece_before = expected_calibration_error(p, y)
        ece_after = expected_calibration_error(_sigmoid(logits / temperature + bias), y)

        fallback = ece_after > ece_before + 1e-12
        if fallback:
            # Minimising NLL usually improves ECE but is not guaranteed to;
            # shipping a calibrator worse than doing nothing is not a trade
            # this system makes.
            log.warning(
                "calibration rejected — NLL optimum did not improve ECE",
                temperature=round(temperature, 4),
                bias=round(bias, 4),
                ece_before=round(ece_before, 4),
                ece_after=round(ece_after, 4),
            )
            temperature, bias = 1.0, 0.0
            ece_after = expected_calibration_error(_sigmoid(logits), y)

        self.temperature, self.bias = float(temperature), float(bias)
        self.report = {
            "temperature": self.temperature,
            "bias": self.bias,
            "n": float(p.size),
            "pos_rate": float(y.mean()),
            "nll_before": _nll(logits, y, 1.0, 0.0),
            "nll_after": _nll(logits, y, self.temperature, self.bias),
            "ece_before": ece_before,
            "ece_after": ece_after,
            "identity_fallback": 1.0 if fallback else 0.0,
        }
        log.info(
            "calibrated",
            temperature=round(self.temperature, 4),
            bias=round(self.bias, 4),
            ece_before=round(ece_before, 4),
            ece_after=round(ece_after, 4),
        )
        return dict(self.report)

    @staticmethod
    def _search(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
        """Coordinate descent on NLL over ``(T, b)``, starting from the identity."""
        temperature, bias = 1.0, 0.0
        grid = np.geomspace(_T_MIN, _T_MAX, _GRID)
        for round_index in range(_ROUNDS):
            if round_index == 0:
                losses = [_nll(logits, labels, float(t), bias) for t in grid]
                best = int(np.argmin(losses))
                low = float(grid[max(0, best - 1)])
                high = float(grid[min(_GRID - 1, best + 1)])
            else:
                low, high = max(_T_MIN, temperature * 0.5), min(_T_MAX, temperature * 2.0)
            # Default args bind the *current* coordinate; the closures are one-shot.
            temperature = _golden_section(
                lambda t, _b=bias: _nll(logits, labels, t, _b), low, high
            )
            bias = _golden_section(
                lambda b, _t=temperature: _nll(logits, labels, _t, b), _B_MIN, _B_MAX
            )
        return temperature, bias

    # -- application ------------------------------------------------------- #

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Apply the fitted ``(T, b)`` to an array of probabilities."""
        return _sigmoid(_logit(np.asarray(probs, dtype=np.float64)) / self.temperature + self.bias)

    def transform_one(self, prob: float) -> float:
        """Scalar convenience used on the request path."""
        return float(self.transform(np.asarray([prob], dtype=np.float64))[0])

    # -- persistence -------------------------------------------------------- #

    def save(self, path: str | Path) -> Path:
        """Write ``{"temperature": ..., "bias": ..., "report": {...}}`` as JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"temperature": self.temperature, "bias": self.bias, "report": self.report}
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> TemperatureCalibrator | None:
        """Read a saved calibrator, or ``None`` when it is absent or unreadable."""
        source = Path(path)
        if not source.exists():
            return None
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
            calibrator = cls(float(payload["temperature"]), float(payload.get("bias", 0.0)))
        except Exception as exc:  # pragma: no cover - corrupt artefact
            log.warning("calibrator unreadable", path=str(source), error=str(exc))
            return None
        calibrator.report = {k: float(v) for k, v in (payload.get("report") or {}).items()}
        return calibrator
