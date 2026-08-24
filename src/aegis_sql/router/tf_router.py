"""The difficulty classifier: **trained in Keras, served in NumPy.**

This split is a deliberate production decision, not a convenience.

Training a small MLP on hand-engineered features is a job Keras does well:
adaptive normalisation, class weighting, early stopping and an AUC metric come
for free, and the whole thing fits in one ``Sequential``.  *Serving* it is a
different problem.  The router runs on the request path, before any generator is
chosen, so it sits in front of 100% of queries — including the ones that will be
answered by the zero-cost template tier in 8 ms.  Importing TensorFlow to
evaluate three matrix multiplies would add ~2 s of process start-up and ~500 MB
of RSS to every API worker, and would make the engine's cheapest path depend on
its heaviest optional dependency.

So the trained parameters are exported to a plain ``.npz`` (the ``Normalization``
mean/variance plus each ``Dense`` kernel/bias) with a small JSON manifest, and
:class:`NumpyRouter` re-implements the forward pass in ~10 lines of NumPy.
**There is no TensorFlow import anywhere in the serving class**, and
``aegis_sql.pipeline`` only ever touches :func:`load_router`.  The round trip is
asserted to agree to ``1e-5`` in ``tests/test_router.py`` — if the two ever
diverge, that is a bug, not a tolerance to widen.

The manifest also pins the feature order.  Retraining after someone adds a
feature to :class:`~aegis_sql.types.DifficultyFeatures` produces weights whose
columns no longer mean what the serving path thinks they mean; the loader
refuses such an artefact instead of silently routing on scrambled inputs.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from aegis_sql.observability.logging import get_logger
from aegis_sql.router.calibrator import roc_auc
from aegis_sql.types import DifficultyFeatures

log = get_logger("router.tf_router")

WEIGHTS_FILE = "router_weights.npz"
META_FILE = "router_meta.json"
KERAS_FILE = "router.keras"

#: Matches ``keras.backend.epsilon()``; the exported manifest carries it so the
#: NumPy path never has to guess.
_NORM_EPSILON = 1e-7
#: Hidden widths of the two ``Dense`` layers; the architecture is fixed because
#: the exported format has to stay readable without TensorFlow.
_HIDDEN = (64, 32)
_DROPOUT = 0.2
_DEFAULT_THRESHOLD = 0.5
_FORMAT_VERSION = 1


# --------------------------------------------------------------------------- #
# Training side (TensorFlow lives strictly inside these methods)
# --------------------------------------------------------------------------- #


class DifficultyRouter:
    """Keras-side owner of the difficulty model: train, evaluate, export.

    Nothing on the request path constructs this class — it exists for
    ``scripts/train_router.py`` and the flywheel.  ``model_dir`` is both the
    training output directory and the directory :class:`NumpyRouter` reads.
    """

    __slots__ = ("model_dir", "model", "metrics", "threshold", "_val_start")

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        #: The Keras model, once :meth:`train` or :meth:`load` has run.
        self.model: Any = None
        self.metrics: dict[str, float] = {}
        self.threshold: float = _DEFAULT_THRESHOLD
        self._val_start: int = 0

    # -- training --------------------------------------------------------- #

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 60,
        val_split: float = 0.2,
        seed: int = 20260824,
    ) -> dict[str, float]:
        """Fit the classifier and return its metrics.

        ``y == 1`` means *the cheap tier failed on this question*, so the
        positive class is "escalation was needed".  That class is the minority
        in any healthy corpus, which is why ``class_weight`` is not optional
        here: without it the model learns to predict "easy" for everything and
        scores 85% accuracy while routing every hard query to a tier that
        cannot answer it.

        Rows are shuffled with a seeded permutation *before* Keras carves off
        its validation split, because ``validation_split`` takes the tail of the
        array as-is — and a corpus written by the flywheel is grouped by
        template, so the untouched tail is not a random sample.
        """
        import tensorflow as tf  # noqa: PLC0415 - heavy, optional, training-only

        random.seed(seed)
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception as exc:  # pragma: no cover - not all TF builds support it
            log.debug("op determinism unavailable", error=str(exc))

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        if X.ndim != 2 or X.shape[0] != y.shape[0]:
            raise ValueError(f"X {X.shape} and y {y.shape} do not line up")
        if X.shape[1] != DifficultyFeatures.dim():
            raise ValueError(f"expected {DifficultyFeatures.dim()} features, got {X.shape[1]}")

        order = np.random.default_rng(seed).permutation(X.shape[0])
        X, y = X[order], y[order]
        self._val_start = int(round(X.shape[0] * (1.0 - val_split)))

        keras = tf.keras
        normalizer = keras.layers.Normalization(axis=-1, name="norm")
        normalizer.adapt(X[: self._val_start] if self._val_start > 1 else X)

        self.model = keras.Sequential(
            [
                keras.layers.Input(shape=(X.shape[1],), name="features"),
                normalizer,
                keras.layers.Dense(_HIDDEN[0], activation="relu", name="dense_0"),
                keras.layers.Dropout(_DROPOUT, name="drop_0"),
                keras.layers.Dense(_HIDDEN[1], activation="relu", name="dense_1"),
                keras.layers.Dense(1, activation="sigmoid", name="head"),
            ],
            name="aegis_difficulty_router",
        )
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=[keras.metrics.AUC(name="auc"), keras.metrics.BinaryAccuracy(name="acc")],
        )

        history = self.model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=32,
            validation_split=val_split,
            class_weight=_class_weight(y),
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=8, restore_best_weights=True, verbose=0
                )
            ],
            verbose=0,
            shuffle=False,  # already permuted; keeps epoch order reproducible
        )

        self.metrics = self._evaluate(X, y, history)
        log.info("router trained", **{k: round(v, 4) for k, v in self.metrics.items()})
        return self.metrics

    def _evaluate(self, X: np.ndarray, y: np.ndarray, history: Any) -> dict[str, float]:
        """Metrics computed from predictions, not scraped from Keras' history.

        ``val_loss`` is the only number taken from the history (it is what early
        stopping restored on); AUC and accuracy are recomputed here so the
        report cannot drift with Keras metric-naming changes.
        """
        probs = self.model.predict(X, verbose=0).ravel().astype(np.float64)
        split = self._val_start
        val_probs, val_y = probs[split:], y[split:]
        hist = getattr(history, "history", {}) or {}
        val_losses = hist.get("val_loss") or [float("nan")]
        return {
            "n": float(X.shape[0]),
            "pos_rate": float(y.mean()),
            "auc": roc_auc(y, probs),
            "acc": float(((probs >= self.threshold).astype(np.float64) == y).mean()),
            "val_auc": roc_auc(val_y, val_probs) if val_y.size else float("nan"),
            "val_acc": (
                float(((val_probs >= self.threshold).astype(np.float64) == val_y).mean())
                if val_y.size
                else float("nan")
            ),
            "val_loss": float(min(val_losses)),
            "epochs_ran": float(len(hist.get("loss", []))),
        }

    def validation_split_index(self) -> int:
        """Row index where the held-out split starts, for calibrating downstream."""
        return self._val_start

    # -- export ----------------------------------------------------------- #

    def export_numpy(self) -> None:
        """Write ``router_weights.npz`` + ``router_meta.json`` for the serving path.

        The artefacts are a pure function of (data, seed, hyper-parameters): no
        timestamps are written, so a re-run with the same inputs produces the
        same bytes and a diff in the model directory means the model actually
        changed.
        """
        if self.model is None:
            raise RuntimeError("train() or load() must run before export_numpy()")

        arrays: dict[str, np.ndarray] = {}
        activations: list[str] = []
        norm_seen = False
        dense_index = 0

        for layer in self.model.layers:
            kind = type(layer).__name__
            if kind == "Normalization":
                arrays["norm_mean"] = np.asarray(layer.mean, dtype=np.float32).reshape(-1)
                arrays["norm_variance"] = np.asarray(layer.variance, dtype=np.float32).reshape(-1)
                norm_seen = True
            elif kind == "Dense":
                kernel, bias = layer.get_weights()
                arrays[f"dense_{dense_index}_kernel"] = np.asarray(kernel, dtype=np.float32)
                arrays[f"dense_{dense_index}_bias"] = np.asarray(bias, dtype=np.float32)
                activations.append(_activation_name(layer))
                dense_index += 1

        if not norm_seen or dense_index == 0:
            raise RuntimeError("model has no Normalization/Dense layers to export")

        self.model_dir.mkdir(parents=True, exist_ok=True)
        np.savez(self.model_dir / WEIGHTS_FILE, **arrays)  # type: ignore[arg-type]
        meta = {
            "format_version": _FORMAT_VERSION,
            "input_dim": int(arrays["norm_mean"].shape[0]),
            "feature_order": list(DifficultyFeatures.ORDER),
            "activations": activations,
            "norm_epsilon": _NORM_EPSILON,
            "threshold": float(self.threshold),
            "metrics": {k: _jsonable(v) for k, v in self.metrics.items()},
        }
        (self.model_dir / META_FILE).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        log.info(
            "router exported",
            dir=str(self.model_dir),
            dense_layers=dense_index,
            params=int(sum(a.size for a in arrays.values())),
        )

    # -- keras-native persistence ---------------------------------------- #

    def save(self) -> None:
        """Persist the Keras model itself (for resuming training, not serving)."""
        if self.model is None:
            raise RuntimeError("nothing to save — train() first")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(self.model_dir / KERAS_FILE)
        log.info("keras model saved", path=str(self.model_dir / KERAS_FILE))

    def load(self) -> bool:
        """Restore a previously saved Keras model.  ``False`` when absent."""
        path = self.model_dir / KERAS_FILE
        if not path.exists():
            return False
        try:
            import tensorflow as tf  # noqa: PLC0415 - training-only dependency

            self.model = tf.keras.models.load_model(path)
        except Exception as exc:  # pragma: no cover - corrupt checkpoint / no TF
            log.warning("keras model unreadable", path=str(path), error=str(exc))
            return False
        meta_path = self.model_dir / META_FILE
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.threshold = float(meta.get("threshold", _DEFAULT_THRESHOLD))
            self.metrics = {k: float(v) for k, v in (meta.get("metrics") or {}).items()}
        return True


# --------------------------------------------------------------------------- #
# Serving side — NumPy only
# --------------------------------------------------------------------------- #


class NumpyRouter:
    """Dependency-free evaluator for the exported difficulty model.

    Forward pass, in order: feature standardisation with the adapted
    mean/variance, ``Dense(64)+ReLU``, ``Dense(32)+ReLU``, ``Dense(1)+sigmoid``.
    Dropout is training-only and therefore absent.  Everything stays ``float32``
    so the arithmetic matches what Keras did rather than merely approximating it.
    """

    __slots__ = ("mean", "scale", "kernels", "biases", "activations", "threshold", "metrics")

    def __init__(
        self,
        mean: np.ndarray,
        variance: np.ndarray,
        kernels: list[np.ndarray],
        biases: list[np.ndarray],
        activations: list[str],
        threshold: float = _DEFAULT_THRESHOLD,
        metrics: dict[str, float] | None = None,
        epsilon: float = _NORM_EPSILON,
    ) -> None:
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, -1)
        # Pre-divide once: Keras uses max(sqrt(var), epsilon) as the denominator.
        self.scale = np.maximum(np.sqrt(np.asarray(variance, dtype=np.float32)), np.float32(epsilon))
        self.scale = self.scale.reshape(1, -1)
        self.kernels = [np.asarray(k, dtype=np.float32) for k in kernels]
        self.biases = [np.asarray(b, dtype=np.float32) for b in biases]
        self.activations = list(activations)
        self.threshold = float(threshold)
        self.metrics = dict(metrics or {})

    # -- construction ------------------------------------------------------ #

    @classmethod
    def load(cls, model_dir: str | Path) -> NumpyRouter | None:
        """Load exported weights, or ``None`` if they are missing or stale."""
        directory = Path(model_dir)
        weights_path = directory / WEIGHTS_FILE
        meta_path = directory / META_FILE
        if not weights_path.exists() or not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            with np.load(weights_path) as npz:
                arrays = {k: npz[k] for k in npz.files}
        except Exception as exc:  # pragma: no cover - corrupt artefact
            log.warning("router artefacts unreadable", dir=str(directory), error=str(exc))
            return None

        order = list(meta.get("feature_order") or [])
        if order != list(DifficultyFeatures.ORDER):
            log.warning(
                "router weights are stale — feature order changed, falling back to the heuristic",
                dir=str(directory),
                expected=len(DifficultyFeatures.ORDER),
                found=len(order),
            )
            return None

        kernels, biases = [], []
        index = 0
        while f"dense_{index}_kernel" in arrays:
            kernels.append(arrays[f"dense_{index}_kernel"])
            biases.append(arrays[f"dense_{index}_bias"])
            index += 1
        if not kernels:
            log.warning("router weights contain no dense layers", dir=str(directory))
            return None

        activations = list(meta.get("activations") or [])
        if len(activations) != len(kernels):  # pragma: no cover - defensive
            activations = ["relu"] * (len(kernels) - 1) + ["sigmoid"]

        router = cls(
            mean=arrays["norm_mean"],
            variance=arrays["norm_variance"],
            kernels=kernels,
            biases=biases,
            activations=activations,
            threshold=float(meta.get("threshold", _DEFAULT_THRESHOLD)),
            metrics={k: float(v) for k, v in (meta.get("metrics") or {}).items() if _is_number(v)},
            epsilon=float(meta.get("norm_epsilon", _NORM_EPSILON)),
        )
        log.info("router loaded", dir=str(directory), auc=round(router.metrics.get("val_auc", 0.0), 4))
        return router

    # -- inference --------------------------------------------------------- #

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """P(hard) for a ``(n, dim)`` matrix, as a ``(n,)`` float64 array."""
        h = (np.asarray(X, dtype=np.float32).reshape(-1, self.mean.shape[1]) - self.mean) / self.scale
        for kernel, bias, activation in zip(self.kernels, self.biases, self.activations, strict=True):
            h = h @ kernel + bias
            if activation == "relu":
                h = np.maximum(h, np.float32(0.0))
            elif activation == "sigmoid":
                h = _sigmoid32(h)
        return h.ravel().astype(np.float64)

    def predict_proba(self, f: DifficultyFeatures | np.ndarray) -> float:
        """P(the cheap tier fails) for a single question."""
        vec = f.to_vector() if isinstance(f, DifficultyFeatures) else f
        return float(self.predict_batch(np.asarray(vec, dtype=np.float32).reshape(1, -1))[0])

    def is_hard(self, f: DifficultyFeatures | np.ndarray) -> bool:
        return self.predict_proba(f) >= self.threshold

    @property
    def input_dim(self) -> int:
        return int(self.mean.shape[1])


def load_router(model_dir: str | Path) -> NumpyRouter | None:
    """Serving-path entry point: exported router, or ``None`` to use the heuristic."""
    try:
        return NumpyRouter.load(model_dir)
    except Exception as exc:  # pragma: no cover - never break the request path
        log.warning("router load failed", dir=str(model_dir), error=str(exc))
        return None


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #


def _sigmoid32(x: np.ndarray) -> np.ndarray:
    """Overflow-free logistic in float32 (``exp(710)`` is ``inf``, and inf/inf is nan)."""
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos], dtype=np.float32))
    exp_neg = np.exp(x[~pos], dtype=np.float32)
    out[~pos] = exp_neg / (1.0 + exp_neg)
    return out


def _class_weight(y: np.ndarray) -> dict[int, float]:
    """Balanced weights; a degenerate single-class corpus falls back to 1.0."""
    n = float(y.size)
    pos = float(y.sum())
    neg = n - pos
    if pos <= 0.0 or neg <= 0.0:
        return {0: 1.0, 1: 1.0}
    return {0: n / (2.0 * neg), 1: n / (2.0 * pos)}


def _activation_name(layer: Any) -> str:
    fn = getattr(layer, "activation", None)
    return getattr(fn, "__name__", "linear")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _jsonable(value: Any) -> Any:
    number = float(value)
    return number if number == number and abs(number) != float("inf") else None
