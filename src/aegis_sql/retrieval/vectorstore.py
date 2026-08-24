"""Pluggable vector storage behind one narrow interface.

Every serious deployment wants a different answer here — one bank standardises
on Chroma, the next has a FAISS sidecar, a laptop demo wants neither — while the
retrieval code above should not care.  So this module defines a four-method
protocol (``add`` / ``search`` / ``persist`` / ``load``) and ships three
interchangeable implementations that agree on their observable behaviour:

* scores are **cosine similarities in ``[-1, 1]``, higher is better** (Chroma's
  cosine *distance* is converted back, so no caller has to remember which
  direction a backend counts in);
* ``where`` is a flat ``{key: value}`` metadata filter where a list/tuple value
  means "any of" — mapped to Chroma's ``$in`` and applied post-hoc by the other
  two;
* metadata values are coerced to Chroma's scalar types (``str | int | float |
  bool``) *everywhere*, so switching backends never changes what comes back.

``persist(path)``/``load(path)`` always take a **directory**; each store writes
files named after its collection inside it.  The numpy store is the default
precisely because it has no dependency, no daemon and no file format anyone has
to migrate: a ``.npz`` of vectors plus a ``.json`` sidecar.
"""

from __future__ import annotations

import importlib.util
import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from aegis_sql.config import Settings
from aegis_sql.observability.logging import get_logger
from aegis_sql.retrieval.embedder import Embedder

log = get_logger("retrieval.vectorstore")

Metadata = dict[str, Any]
SearchHit = tuple[str, float, Metadata]

_CHROMA_NAME_RE = re.compile(r"[^a-zA-Z0-9._-]")


@runtime_checkable
class VectorStore(Protocol):
    """Minimal vector index contract used by schema linking and few-shot search."""

    name: str

    def add(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        vectors: np.ndarray,
        metadatas: Sequence[Metadata] | None = None,
    ) -> None:
        """Insert or replace ``ids``; ``vectors`` is a ``(n, dim)`` float matrix."""
        ...

    def search(self, vector: np.ndarray, k: int = 10, where: Metadata | None = None) -> list[SearchHit]:
        """Return up to ``k`` ``(id, cosine_score, metadata)`` triples, best first."""
        ...

    def persist(self, path: str | Path) -> None:
        """Write the collection into the directory ``path``."""
        ...

    def load(self, path: str | Path) -> None:
        """Replace the in-memory collection with the one stored in ``path``."""
        ...

    def __len__(self) -> int:
        ...


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _clean_metadata(md: Metadata | None) -> Metadata:
    """Coerce to the scalar types Chroma accepts so backends stay interchangeable."""
    out: Metadata = {}
    for key, value in (md or {}).items():
        if value is None:
            out[str(key)] = ""
        elif isinstance(value, (str, int, float, bool)):
            out[str(key)] = value
        elif isinstance(value, (list, tuple, set)):
            out[str(key)] = ",".join(str(v) for v in value)
        else:
            out[str(key)] = str(value)
    return out


def _matches(md: Metadata, where: Metadata | None) -> bool:
    if not where:
        return True
    for key, wanted in where.items():
        actual = md.get(key)
        if isinstance(wanted, (list, tuple, set)):
            if actual not in wanted:
                return False
        elif actual != wanted:
            return False
    return True


def _to_chroma_where(where: Metadata | None) -> Metadata | None:
    if not where:
        return None
    out: Metadata = {}
    for key, wanted in where.items():
        out[key] = {"$in": list(wanted)} if isinstance(wanted, (list, tuple, set)) else wanted
    return out


def _normalise(vectors: np.ndarray) -> np.ndarray:
    mat = np.asarray(vectors, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, 1e-12)


def _sanitize_collection(name: str) -> str:
    """Chroma requires 3-512 chars of ``[a-zA-Z0-9._-]`` starting/ending alnum."""
    cleaned = _CHROMA_NAME_RE.sub("-", name).strip("-._")
    if not cleaned:
        cleaned = "aegis"
    if not cleaned[0].isalnum():
        cleaned = f"a{cleaned}"
    while len(cleaned) < 3:
        cleaned += "0"
    return cleaned[:512]


def _top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` largest scores, sorted descending (ties by index)."""
    k = min(k, scores.shape[0])
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part], kind="stable")]


# --------------------------------------------------------------------------- #
# numpy (default)
# --------------------------------------------------------------------------- #


class NumpyVectorStore:
    """Exhaustive cosine search over a dense matrix.

    At the scale schema linking actually runs at — a few hundred column
    documents, a few thousand few-shot examples — a brute-force matmul is faster
    than any ANN index and returns exact neighbours, which keeps evaluation
    numbers attributable to the retriever rather than to index recall.
    """

    __slots__ = ("name", "_ids", "_texts", "_meta", "_matrix", "_pos")

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._meta: list[Metadata] = []
        self._matrix: np.ndarray | None = None
        self._pos: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def dim(self) -> int:
        return 0 if self._matrix is None else int(self._matrix.shape[1])

    def add(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        vectors: np.ndarray,
        metadatas: Sequence[Metadata] | None = None,
    ) -> None:
        mat = _normalise(vectors)
        if len(ids) != mat.shape[0] or len(ids) != len(texts):
            raise ValueError("ids, texts and vectors must have the same length")
        metas = [_clean_metadata(m) for m in (metadatas or [{} for _ in ids])]

        fresh_rows: list[np.ndarray] = []
        for i, doc_id in enumerate(ids):
            existing = self._pos.get(doc_id)
            if existing is not None and self._matrix is not None:
                self._texts[existing] = texts[i]
                self._meta[existing] = metas[i]
                self._matrix[existing] = mat[i]
                continue
            self._pos[doc_id] = len(self._ids)
            self._ids.append(doc_id)
            self._texts.append(texts[i])
            self._meta.append(metas[i])
            fresh_rows.append(mat[i])
        if fresh_rows:
            block = np.vstack(fresh_rows).astype(np.float32)
            self._matrix = block if self._matrix is None else np.vstack([self._matrix, block])

    def search(self, vector: np.ndarray, k: int = 10, where: Metadata | None = None) -> list[SearchHit]:
        if self._matrix is None or not self._ids:
            return []
        query = _normalise(vector)[0]
        scores = self._matrix @ query
        if where:
            keep = np.array([_matches(m, where) for m in self._meta], dtype=bool)
            scores = np.where(keep, scores, -np.inf)
        hits: list[SearchHit] = []
        for idx in _top_k(scores, k):
            score = float(scores[idx])
            if not np.isfinite(score):
                continue
            hits.append((self._ids[idx], score, dict(self._meta[idx])))
        return hits

    def document(self, doc_id: str) -> str | None:
        pos = self._pos.get(doc_id)
        return None if pos is None else self._texts[pos]

    # -- persistence -------------------------------------------------------- #

    def persist(self, path: str | Path) -> None:
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        matrix = self._matrix if self._matrix is not None else np.zeros((0, 0), dtype=np.float32)
        np.savez_compressed(directory / f"{self.name}.npz", vectors=matrix)
        payload = {"name": self.name, "ids": self._ids, "texts": self._texts, "metadatas": self._meta}
        (directory / f"{self.name}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        log.debug("vector store persisted", backend="numpy", name=self.name, n=len(self._ids))

    def load(self, path: str | Path) -> None:
        directory = Path(path)
        payload = json.loads((directory / f"{self.name}.json").read_text(encoding="utf-8"))
        with np.load(directory / f"{self.name}.npz") as data:
            matrix = data["vectors"].astype(np.float32)
        self._ids = list(payload["ids"])
        self._texts = list(payload["texts"])
        self._meta = [dict(m) for m in payload["metadatas"]]
        self._matrix = matrix if matrix.size else None
        self._pos = {doc_id: i for i, doc_id in enumerate(self._ids)}


# --------------------------------------------------------------------------- #
# chroma
# --------------------------------------------------------------------------- #


class ChromaVectorStore:
    """Chroma backend.  Ephemeral until :meth:`persist` names a directory.

    ``embedding_function=None`` is deliberate: AEGIS-SQL always supplies its own
    vectors, and Chroma's default function would try to download an ONNX model
    the first time a collection is created.
    """

    __slots__ = ("name", "_collection", "_client", "_dir")

    def __init__(self, name: str = "default", persist_dir: str | Path | None = None) -> None:
        self.name = _sanitize_collection(name)
        self._dir: Path | None = Path(persist_dir) if persist_dir else None
        self._client, self._collection = self._open(self._dir)

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("chromadb") is not None

    def _open(self, persist_dir: Path | None) -> tuple[Any, Any]:
        import chromadb  # lazy

        if persist_dir is not None:
            persist_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(persist_dir))
        else:
            client = chromadb.EphemeralClient()
        collection = client.get_or_create_collection(
            self.name, embedding_function=None, metadata={"hnsw:space": "cosine"}
        )
        return client, collection

    def __len__(self) -> int:
        return int(self._collection.count())

    def add(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        vectors: np.ndarray,
        metadatas: Sequence[Metadata] | None = None,
    ) -> None:
        mat = _normalise(vectors)
        metas = [_clean_metadata(m) for m in (metadatas or [{} for _ in ids])]
        # Chroma rejects empty metadata dicts on some versions; a marker keeps
        # the payload valid without inventing a filterable field.
        metas = [m or {"_": ""} for m in metas]
        self._collection.upsert(
            ids=list(ids), embeddings=mat.tolist(), documents=list(texts), metadatas=metas
        )

    def search(self, vector: np.ndarray, k: int = 10, where: Metadata | None = None) -> list[SearchHit]:
        count = len(self)
        if count == 0:
            return []
        result = self._collection.query(
            query_embeddings=_normalise(vector).tolist(),
            n_results=min(k, count),
            where=_to_chroma_where(where),
            include=["distances", "metadatas"],
        )
        ids = (result.get("ids") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        hits: list[SearchHit] = []
        for i, doc_id in enumerate(ids):
            # Cosine space: distance = 1 - similarity.
            score = 1.0 - float(dists[i])
            hits.append((doc_id, score, dict(metas[i] or {})))
        return hits

    def persist(self, path: str | Path) -> None:
        """Write a *copy* into ``path``; this instance keeps serving from where it is.

        Migrating the live handle instead would make the store die with the
        directory it was snapshotted into, which is not what "persist" means for
        the other two backends.
        """
        directory = Path(path)
        if self._dir is not None and directory.resolve() == self._dir.resolve():
            return  # PersistentClient already writes through
        rows = self._collection.get(include=["embeddings", "documents", "metadatas"])
        _client, collection = self._open(directory)
        ids = list(rows.get("ids") or [])
        if ids:
            embeddings = np.asarray(rows["embeddings"], dtype=np.float32)
            collection.upsert(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=list(rows.get("documents") or ["" for _ in ids]),
                metadatas=[dict(m or {"_": ""}) for m in (rows.get("metadatas") or [{} for _ in ids])],
            )
        log.debug("vector store persisted", backend="chroma", name=self.name, n=len(ids))

    def load(self, path: str | Path) -> None:
        self._dir = Path(path)
        self._client, self._collection = self._open(self._dir)


# --------------------------------------------------------------------------- #
# faiss
# --------------------------------------------------------------------------- #


class FaissVectorStore:
    """FAISS ``IndexFlatIP`` over L2-normalised vectors (inner product == cosine).

    The id→row mapping and payloads live in Python; the index is rebuilt on
    upsert rather than mutated, because ``IndexFlat`` has no in-place update and
    a rebuild of a few thousand rows costs microseconds.
    """

    __slots__ = ("name", "_ids", "_texts", "_meta", "_matrix", "_index", "_pos")

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._meta: list[Metadata] = []
        self._matrix: np.ndarray | None = None
        self._index: Any = None
        self._pos: dict[str, int] = {}

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("faiss") is not None

    def __len__(self) -> int:
        return len(self._ids)

    def _rebuild(self) -> None:
        import faiss  # lazy

        if self._matrix is None or self._matrix.size == 0:
            self._index = None
            return
        index = faiss.IndexFlatIP(int(self._matrix.shape[1]))
        index.add(np.ascontiguousarray(self._matrix, dtype=np.float32))
        self._index = index

    def add(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        vectors: np.ndarray,
        metadatas: Sequence[Metadata] | None = None,
    ) -> None:
        mat = _normalise(vectors)
        if len(ids) != mat.shape[0] or len(ids) != len(texts):
            raise ValueError("ids, texts and vectors must have the same length")
        metas = [_clean_metadata(m) for m in (metadatas or [{} for _ in ids])]
        rows = list(self._matrix) if self._matrix is not None else []
        for i, doc_id in enumerate(ids):
            existing = self._pos.get(doc_id)
            if existing is not None:
                self._texts[existing] = texts[i]
                self._meta[existing] = metas[i]
                rows[existing] = mat[i]
                continue
            self._pos[doc_id] = len(self._ids)
            self._ids.append(doc_id)
            self._texts.append(texts[i])
            self._meta.append(metas[i])
            rows.append(mat[i])
        self._matrix = np.vstack(rows).astype(np.float32) if rows else None
        self._rebuild()

    def search(self, vector: np.ndarray, k: int = 10, where: Metadata | None = None) -> list[SearchHit]:
        if self._index is None or not self._ids:
            return []
        # Over-fetch so that post-filtering still returns k survivors.
        fetch = min(len(self._ids), k * 5 if where else k)
        scores, idxs = self._index.search(np.ascontiguousarray(_normalise(vector)), fetch)
        hits: list[SearchHit] = []
        for score, idx in zip(scores[0], idxs[0], strict=False):
            if idx < 0:
                continue
            meta = self._meta[int(idx)]
            if not _matches(meta, where):
                continue
            hits.append((self._ids[int(idx)], float(score), dict(meta)))
            if len(hits) >= k:
                break
        return hits

    def persist(self, path: str | Path) -> None:
        import faiss  # lazy

        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            faiss.write_index(self._index, str(directory / f"{self.name}.faiss"))
        payload = {"name": self.name, "ids": self._ids, "texts": self._texts, "metadatas": self._meta}
        (directory / f"{self.name}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        log.debug("vector store persisted", backend="faiss", name=self.name, n=len(self._ids))

    def load(self, path: str | Path) -> None:
        import faiss  # lazy

        directory = Path(path)
        payload = json.loads((directory / f"{self.name}.json").read_text(encoding="utf-8"))
        self._ids = list(payload["ids"])
        self._texts = list(payload["texts"])
        self._meta = [dict(m) for m in payload["metadatas"]]
        self._pos = {doc_id: i for i, doc_id in enumerate(self._ids)}
        index_path = directory / f"{self.name}.faiss"
        if index_path.exists():
            self._index = faiss.read_index(str(index_path))
            self._matrix = self._index.reconstruct_n(0, self._index.ntotal).astype(np.float32)
        else:  # pragma: no cover - empty collection
            self._index, self._matrix = None, None


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #


def get_vector_store(settings: Settings, name: str = "default") -> VectorStore:
    """Instantiate the backend named by ``settings.retrieval.vector_store``.

    ``"auto"`` prefers Chroma when it is importable and falls back to the numpy
    store, so the engine has a working index in every environment.  Nothing is
    written to disk until :meth:`VectorStore.persist` is called.
    """
    choice = (settings.retrieval.vector_store or "auto").strip().lower()
    if choice in {"numpy", "memory", "inmemory"}:
        return NumpyVectorStore(name)
    if choice == "faiss":
        if FaissVectorStore.available():
            return FaissVectorStore(name)
        log.warning("faiss requested but not importable; using numpy store", name=name)
        return NumpyVectorStore(name)
    if choice == "chroma" or choice == "auto":
        if ChromaVectorStore.available():
            try:
                return ChromaVectorStore(name)
            except Exception as exc:  # pragma: no cover - broken chroma install
                log.warning("chroma unavailable; using numpy store", error=str(exc)[:120])
        elif choice == "chroma":
            log.warning("chroma requested but not importable; using numpy store", name=name)
        return NumpyVectorStore(name)
    log.warning("unknown vector_store setting; using numpy store", requested=choice)
    return NumpyVectorStore(name)


def build_store(
    settings: Settings,
    name: str,
    documents: Iterable[tuple[str, str, Metadata]],
    embedder: Embedder,
) -> VectorStore:
    """Convenience: embed ``(id, text, metadata)`` triples into a fresh store."""
    triples = list(documents)
    store = get_vector_store(settings, name)
    if not triples:
        return store
    ids = [t[0] for t in triples]
    texts = [t[1] for t in triples]
    metas = [t[2] for t in triples]
    store.add(ids, texts, embedder.encode(texts), metas)
    return store
