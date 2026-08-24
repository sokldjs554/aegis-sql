"""Text embedding for schema and few-shot retrieval.

The retriever has to work on two very awkward kinds of text at once: Korean
logical names written without whitespace (``계약상태코드``) and cryptic ASCII
physical names (``CTRT_STAT_CD``).  A word-level model fails on both — Korean
because there are no spaces to tokenise on, ASCII because ``CTRT`` is not a word
in any pretrained vocabulary.

So the default embedder here is a *hashing* embedder over character n-grams
(n=2,3) plus underscore-split word unigrams.  It has three properties that
matter more for this system than raw semantic quality:

* **Zero model downloads.**  The engine must boot in an air-gapped bank VPC.
  A sentence-transformers backend is used only when the weights already sit in
  the local HF cache; it is never fetched at runtime.
* **Determinism.**  Buckets come from FNV-1a, not Python's per-process
  randomised ``hash()``, so an index built today matches one built tomorrow and
  evaluation numbers are reproducible.
* **Sub-word recall on Korean.**  ``지점별`` and ``지점명`` share the bigram
  ``지점``, which is exactly the signal that links a question to ``TB_BRCH``.

Signed hashing (Weinberger et al., 2009) keeps collisions unbiased: two features
landing in the same bucket cancel in expectation instead of always adding.
Term weights use sublinear TF (``1 + log tf``) and an optional per-bucket IDF
fitted on the indexed corpus, which is the hashing-trick approximation of
TF-IDF — the vocabulary is never materialised, so memory is O(dim), not
O(|vocab|).
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from aegis_sql.config import Settings
from aegis_sql.observability.logging import get_logger

log = get_logger("retrieval.embedder")

#: Eojeol-level tokens: ASCII identifier runs (``ctrt_stat_cd``) or Hangul runs.
_WORD_RE = re.compile(r"[a-z0-9_]+|[가-힣]+")
_HANGUL_RE = re.compile(r"[가-힣]")

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_MASK64 = 0xFFFFFFFFFFFFFFFF


@lru_cache(maxsize=200_000)
def _fnv1a(token: str) -> int:
    """64-bit FNV-1a — a stable substitute for the randomised builtin ``hash``."""
    h = _FNV_OFFSET
    for byte in token.encode("utf-8"):
        h = ((h ^ byte) * _FNV_PRIME) & _MASK64
    return h


def _features(text: str) -> Counter[str]:
    """Bag of hashable features: word unigrams, underscore sub-words, char n-grams."""
    feats: Counter[str] = Counter()
    for word in _WORD_RE.findall(text.lower()):
        feats[f"w:{word}"] += 1
        # ASCII identifiers carry their meaning in the underscore-separated
        # segments; Hangul runs carry it in the characters themselves.
        units = word.split("_") if "_" in word else [word]
        if len(units) > 1:
            for unit in units:
                if unit:
                    feats[f"w:{unit}"] += 1
        for unit in units:
            for n in (2, 3):
                for i in range(len(unit) - n + 1):
                    feats[f"c{n}:{unit[i : i + n]}"] += 1
    return feats


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns text into L2-normalised row vectors."""

    dim: int
    name: str

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return a ``(len(texts), dim)`` float32 matrix with unit-norm rows."""
        ...


class HashingEmbedder:
    """Dependency-free character n-gram hashing embedder.

    ``fit`` learns a per-bucket IDF from the corpus that will be searched;
    ``fitted`` returns an independently fitted copy, which is what callers that
    share one embedder instance across sub-systems should use (fitting in place
    would silently invalidate a matrix another component already encoded).
    """

    __slots__ = ("dim", "name", "_idf")

    def __init__(self, dim: int = 256, idf: np.ndarray | None = None) -> None:
        if dim < 16:
            raise ValueError("embedding_dim must be >= 16")
        self.dim = int(dim)
        self.name = f"hashing-{self.dim}"
        self._idf: np.ndarray | None = None
        if idf is not None:
            self._idf = np.asarray(idf, dtype=np.float32).reshape(self.dim)

    # -- fitting ----------------------------------------------------------- #

    def _document_frequency(self, corpus: Sequence[str]) -> np.ndarray:
        df = np.zeros(self.dim, dtype=np.float64)
        for text in corpus:
            buckets = {_fnv1a(f) % self.dim for f in _features(text)}
            for b in buckets:
                df[b] += 1.0
        return df

    def fit(self, corpus: Sequence[str]) -> HashingEmbedder:
        """Fit the per-bucket IDF **in place** (sklearn convention) and return self."""
        n_docs = max(1, len(corpus))
        df = self._document_frequency(corpus)
        self._idf = (np.log((n_docs + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)
        return self

    def fitted(self, corpus: Sequence[str]) -> HashingEmbedder:
        """Return a *copy* fitted on ``corpus``, leaving this instance untouched."""
        clone = HashingEmbedder(dim=self.dim)
        return clone.fit(corpus)

    @property
    def is_fitted(self) -> bool:
        return self._idf is not None

    # -- encoding ---------------------------------------------------------- #

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        idf = self._idf
        for row, text in enumerate(texts):
            for feat, tf in _features(text).items():
                h = _fnv1a(feat)
                bucket = h % self.dim
                # Bucket comes from the low bits, sign from bit 32: independent
                # draws, so collisions cancel rather than accumulate.
                sign = 1.0 if (h >> 32) & 1 else -1.0
                weight = 1.0 + math.log(tf)
                if idf is not None:
                    weight *= float(idf[bucket])
                out[row, bucket] += sign * weight
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        np.divide(out, np.maximum(norms, 1e-12), out=out)
        return out


class SentenceTransformerEmbedder:
    """Optional dense backend — used only when the weights are already cached.

    Constructing this class never touches the network: the local cache is probed
    on the filesystem first, and the model is then loaded with
    ``local_files_only=True``.  Any failure raises so that :func:`get_embedder`
    can fall back to :class:`HashingEmbedder`.
    """

    __slots__ = ("dim", "name", "_model", "_batch_size")

    def __init__(
        self,
        model_name: str,
        cache_folder: str | Path | None = None,
        batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        local = _local_model_path(model_name, cache_folder)
        if local is None:
            raise FileNotFoundError(
                f"sentence-transformers model '{model_name}' is not in the local cache; "
                "AEGIS-SQL never downloads weights at runtime"
            )
        from sentence_transformers import SentenceTransformer  # lazy: pulls torch

        self._model = SentenceTransformer(
            str(local),
            device=device,
            local_files_only=True,
            cache_folder=str(cache_folder) if cache_folder else None,
        )
        self._batch_size = batch_size
        self.dim = int(self._model.get_sentence_embedding_dimension() or 0)
        self.name = f"st:{model_name}"

    @classmethod
    def try_load(cls, model_name: str, **kwargs: object) -> SentenceTransformerEmbedder | None:
        try:
            return cls(model_name, **kwargs)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover - environment dependent
            log.info(
                "sentence-transformers unavailable, falling back",
                model=model_name,
                reason=str(exc)[:120],
            )
            return None

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)


def _local_model_path(model_name: str, cache_folder: str | Path | None = None) -> Path | None:
    """Locate a cached HF/ST model without importing (or contacting) the hub."""
    direct = Path(model_name)
    if direct.is_dir():
        return direct

    roots: list[Path] = []
    if cache_folder:
        roots.append(Path(cache_folder))
    for env in ("SENTENCE_TRANSFORMERS_HOME", "HF_HOME", "HUGGINGFACE_HUB_CACHE"):
        value = os.environ.get(env)
        if value:
            roots.extend([Path(value), Path(value) / "hub"])
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    roots.append(Path.home() / ".cache" / "torch" / "sentence_transformers")

    stems = [
        f"models--{model_name.replace('/', '--')}",
        model_name.replace("/", "_"),
        model_name.split("/")[-1],
    ]
    for root in roots:
        for stem in stems:
            candidate = root / stem
            if candidate.is_dir():
                # HF hub layout keeps the actual files under snapshots/<sha>/.
                snapshots = candidate / "snapshots"
                if snapshots.is_dir():
                    snaps = sorted(p for p in snapshots.iterdir() if p.is_dir())
                    return snaps[-1] if snaps else None
                return candidate
    return None


def get_embedder(settings: Settings) -> Embedder:
    """Build the embedder named by ``settings.retrieval.embedder``.

    ``"auto"`` prefers a locally cached sentence-transformers model and silently
    degrades to the hashing embedder, which is the configuration the repository
    ships with (and the one every test runs against).
    """
    cfg = settings.retrieval
    choice = (cfg.embedder or "auto").strip().lower()
    dense_aliases = {"sentence-transformers", "sentence_transformers", "st", "sbert", "dense"}

    if choice in {"hashing", "hash", "numpy", "sparse"}:
        return HashingEmbedder(dim=cfg.embedding_dim)

    emb = SentenceTransformerEmbedder.try_load(cfg.embedding_model)
    if emb is not None:
        log.info("embedder selected", backend=emb.name, dim=emb.dim)
        return emb
    if choice in dense_aliases:
        log.warning("requested dense embedder is unavailable offline", model=cfg.embedding_model)
    fallback = HashingEmbedder(dim=cfg.embedding_dim)
    log.info("embedder selected", backend=fallback.name, dim=fallback.dim)
    return fallback
