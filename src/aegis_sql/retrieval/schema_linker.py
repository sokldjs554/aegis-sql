"""Hybrid schema linking: pruning the schema down to what the question needs.

The demo schema costs ~2,550 prompt tokens.  A real Korean insurance core has
several hundred tables, so "just paste the DDL" stops being an option long
before it stops being tempting: the prompt blows past the context window, and
long before that the model starts joining the wrong ``*_CD`` column.  Schema
linking is the retrieval step that keeps the prompt small *and* accurate.

Pure dense retrieval is not enough here, and the failure is structural rather
than a matter of model quality.  ``CTRT_STAT_CD`` is not a word; ``실효`` never
appears in the schema text at all (it lives in ``TB_COMM_CD`` as the *label* of
code ``'02'``).  So five signals are fused per column:

* **dense**   — embedding similarity over one document per table and per column;
* **lexical** — BM25 (k1=1.2, b=0.75) over the same documents, implemented here
  so the engine keeps no search-server dependency.  Korean is indexed as
  character bigrams, which is what makes 지점``별`` match 지점``명``;
* **glossary** — the business dictionary, injecting 실효 → ``CTRT_STAT_CD``;
* **value**  — the question's literals matched against *profiled* column values
  and code labels, so 실효 lands on the column that actually stores it;
* **structure** — FK expansion plus a Steiner-tree join path, because a column
  that is reachable only through a bridge table (``TB_CTRT → TB_AGNT →
  TB_BRCH``) is useless without the bridge.

Recall is the metric that matters: a dropped table is an unrecoverable error,
while an extra table costs a few dozen tokens.  Hence the deliberate
asymmetries — join-path and primary-key columns are force-included, a
``min_tables`` floor overrides the score, and ``LinkedSchema.coverage`` reports
how aggressive the pruning was so the router can escalate when it looks unsafe.

Every selected object carries its evidence (:class:`~aegis_sql.types.ScoredItem`)
so the API can answer "why is this column in my prompt?" — an audit requirement
in regulated deployments, and the fastest debugging tool during development.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from aegis_sql.config import Settings
from aegis_sql.observability.logging import get_logger
from aegis_sql.retrieval.embedder import Embedder
from aegis_sql.retrieval.glossary import Glossary, GlossaryMatch
from aegis_sql.schema.graph import JoinEdge, JoinGraph
from aegis_sql.schema.profile import ColumnProfile, SchemaProfile
from aegis_sql.types import (
    ForeignKey,
    LinkedSchema,
    NormalizedQuestion,
    SchemaGraph,
    ScoredItem,
)

log = get_logger("retrieval.schema_linker")

BM25_K1 = 1.2
BM25_B = 0.75
#: Weight of a table's own document score when ranking tables by their columns.
TABLE_PRIOR = 0.15
#: A glossary hit on a *table* is weaker evidence than a hit on a column.
GLOSSARY_TABLE_SHARE = 0.6
#: A column seeds its table only above this fraction of the best column score.
SEED_RATIO = 0.45
#: FK-reachable tables need weaker evidence than seeds, but not none.
FK_ADMIT_RATIO = 0.3
#: Below this score a glossary hit informs the prompt but does not pull in a table.
GLOSSARY_SEED_FLOOR = 0.5
#: Sample values above these bounds are free text, not categories — see
#: :meth:`SchemaLinker._column_document`.
MAX_INDEXED_VALUE_CHARS = 32
MAX_INDEXED_VALUE_WORDS = 3.0
#: A contributing signal must carry at least this much of the total to be cited.
SOURCE_EPSILON = 0.05
MAX_EVIDENCE = 20

_WORD_RE = re.compile(r"[a-z0-9_]+|[가-힣]+")


@dataclass(slots=True)
class LinkCandidate:
    """One scored schema object (``TB_CTRT`` or ``TB_CTRT.MON_PRM``)."""

    ref: str
    dense: float = 0.0
    lexical: float = 0.0
    glossary: float = 0.0
    value: float = 0.0
    total: float = 0.0
    sources: list[str] = field(default_factory=list)

    @property
    def table(self) -> str:
        return self.ref.split(".", 1)[0]

    @property
    def is_column(self) -> bool:
        return "." in self.ref


@dataclass(slots=True)
class _Index:
    """Immutable per-schema retrieval index, cached by schema fingerprint."""

    refs: list[str]
    docs: list[str]
    matrix: np.ndarray
    embedder: Embedder
    #: BM25 statistics over ``docs``: term -> [(doc row, term frequency), ...].
    postings: dict[str, list[tuple[int, int]]]
    doc_lens: list[int]
    avg_len: float
    idf: dict[str, float]
    #: ``ref`` → searchable value/label strings, for value linking.
    value_index: dict[str, tuple[frozenset[str], frozenset[str], bool]]
    row_of: dict[str, int]


#: Building the index costs an embedding pass over every column; the pipeline
#: constructs a linker per request, so the work is shared process-wide.
_INDEX_CACHE: dict[str, _Index] = {}


class SchemaLinker:
    """Scores every table and column against a question and prunes the schema."""

    def __init__(
        self,
        schema: SchemaGraph,
        profile: SchemaProfile | None,
        join_graph: JoinGraph,
        glossary: Glossary,
        embedder: Embedder,
        settings: Settings,
    ) -> None:
        self.schema = schema
        self.profile = profile
        self.join_graph = join_graph
        self.glossary = glossary
        self.embedder = embedder
        self.settings = settings
        self._index: _Index | None = None

    # -- indexing ---------------------------------------------------------- #

    @property
    def cache_key(self) -> str:
        """Identifies the index: schema shape, embedder identity and profile version."""
        dim = getattr(self.embedder, "dim", 0)
        name = getattr(self.embedder, "name", "embedder")
        profile_fp = self.profile.fingerprint if self.profile else "noprofile"
        return f"{self.schema.fingerprint()}:{name}:{dim}:{profile_fp}"

    def build_index(self) -> None:
        """Embed one document per table and per column.  Idempotent and cached."""
        if self._index is not None:
            return
        cached = _INDEX_CACHE.get(self.cache_key)
        if cached is not None:
            self._index = cached
            return

        refs: list[str] = []
        docs: list[str] = []
        for table in self.schema.tables.values():
            refs.append(table.name)
            docs.append(self._table_document(table.name))
            for col in table.columns:
                refs.append(col.qualified)
                docs.append(self._column_document(table.name, col.name))

        # Fit IDF on a *copy* so a shared embedder instance is never mutated
        # underneath another sub-system that already encoded with it.
        embedder = self.embedder
        fitted = getattr(embedder, "fitted", None)
        if callable(fitted):
            embedder = fitted(docs)
        matrix = embedder.encode(docs)

        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        doc_lens: list[int] = []
        for row, doc in enumerate(docs):
            tf = _term_frequencies(doc)
            doc_lens.append(sum(tf.values()))
            for term, freq in tf.items():
                postings[term].append((row, freq))
        avg_len = (sum(doc_lens) / len(doc_lens)) if doc_lens else 0.0
        n_docs = max(1, len(docs))
        idf = {
            term: math.log(1.0 + (n_docs - len(rows) + 0.5) / (len(rows) + 0.5))
            for term, rows in postings.items()
        }

        index = _Index(
            refs=refs,
            docs=docs,
            matrix=matrix,
            embedder=embedder,
            postings=dict(postings),
            doc_lens=doc_lens,
            avg_len=avg_len or 1.0,
            idf=idf,
            value_index=self._build_value_index(),
            row_of={ref: i for i, ref in enumerate(refs)},
        )
        _INDEX_CACHE[self.cache_key] = index
        self._index = index
        log.info(
            "schema index built",
            documents=len(docs),
            tables=len(self.schema.tables),
            embedder=getattr(embedder, "name", "?"),
        )

    def document(self, ref: str) -> str | None:
        """The indexed text for ``TB_CTRT`` or ``TB_CTRT.MON_PRM`` — explainability hook."""
        if self._index is None:
            return None
        row = self._index.row_of.get(ref)
        return None if row is None else self._index.docs[row]

    def _table_document(self, table_name: str) -> str:
        table = self.schema.table(table_name)
        if table is None:  # pragma: no cover - defensive
            return table_name
        head = f"{table.name} {table.comment or ''}".strip()
        cols = " ".join(f"{c.name} {c.comment or ''}".strip() for c in table.columns)
        return f"{head} | 테이블 | {cols}"

    def _column_document(self, table_name: str, column_name: str) -> str:
        """``TB_CTRT 계약 | CTRT_STAT_CD 계약상태코드 TEXT 코드값: 01=정상,02=실효``.

        Sample values are what let a question's literals find their column, but
        they are also how value-augmented retrieval poisons itself: profiling a
        free-text column (``CNTN 상담내용``) on a small sample makes it look
        low-cardinality, and its complaint bodies then contain every domain word
        in the language — turning it into the top BM25 hit for any question
        about 보험금 지급.  Long values are therefore never indexed.
        """
        table = self.schema.table(table_name)
        col = self.schema.column(table_name, column_name)
        if table is None or col is None:  # pragma: no cover - defensive
            return f"{table_name}.{column_name}"
        parts = [f"{table.name} {table.comment or ''}".strip(), "|"]
        parts.append(f"{col.name} {col.comment or ''} {col.dtype}".strip())
        cp = self.profile.get(table_name, column_name) if self.profile else None
        if cp and cp.code_labels:
            pairs = ",".join(f"{k}={v}" for k, v in sorted(cp.code_labels.items())[:12])
            parts.append(f"코드값: {pairs}")
        elif cp and cp.is_yyyymmdd:
            parts.append(f"YYYYMMDD 일자 {cp.min_value}~{cp.max_value}")
        elif cp and _is_indexable_category(cp):
            parts.append("값: " + ",".join(cp.values[:10]))
        if col.foreign_key:
            parts.append(f"FK {col.foreign_key.to_table}.{col.foreign_key.to_column}")
        return " ".join(p for p in parts if p)

    def _build_value_index(self) -> dict[str, tuple[frozenset[str], frozenset[str], bool]]:
        """``ref`` → (profiled values, code labels + codes, is_categorical)."""
        out: dict[str, tuple[frozenset[str], frozenset[str], bool]] = {}
        if not self.profile:
            return out
        for col in self.schema.all_columns:
            cp = self.profile.get(col.table, col.name)
            if not cp:
                continue
            values = frozenset(v.lower() for v in cp.values if v)
            labels = frozenset(
                {v.lower() for v in cp.code_labels.values() if v} | {k.lower() for k in cp.code_labels}
            )
            if values or labels:
                out[col.qualified] = (values, labels, _is_indexable_category(cp))
        return out

    # -- linking ------------------------------------------------------------ #

    def link(self, nq: NormalizedQuestion) -> LinkedSchema:
        """Score, prune and connect: the question's sub-schema plus its evidence."""
        self.build_index()
        index = self._index
        assert index is not None  # build_index always populates it
        cfg = self.settings.retrieval

        text = nq.normalized or nq.raw
        tokens = [t for t in (nq.tokens or _WORD_RE.findall(text.lower())) if t]

        dense = _normalise_signal(index.matrix @ index.embedder.encode([text])[0])
        lexical = _normalise_signal(self._bm25(text, tokens, index))

        matches = self.glossary.match(tokens, text)
        gloss_boost = self._glossary_boosts(matches)
        value_boost = self._value_boosts(nq, tokens, matches, index)

        dw = float(cfg.dense_weight)
        candidates: dict[str, LinkCandidate] = {}
        for i, ref in enumerate(index.refs):
            cand = LinkCandidate(
                ref=ref,
                dense=float(dense[i]),
                lexical=float(lexical[i]),
                glossary=float(gloss_boost.get(ref, 0.0)),
                value=float(value_boost.get(ref, 0.0)),
            )
            weighted = {
                "dense": dw * cand.dense,
                "lexical": (1.0 - dw) * cand.lexical,
                "glossary": float(cfg.glossary_weight) * cand.glossary,
                "value": float(cfg.value_match_weight) * cand.value,
            }
            cand.total = round(sum(weighted.values()), 6)
            cand.sources = [k for k, v in weighted.items() if v > SOURCE_EPSILON]
            candidates[ref] = cand

        columns = sorted(
            (c for c in candidates.values() if c.is_column), key=lambda c: (-c.total, c.ref)
        )
        top_columns = columns[: max(1, int(cfg.top_k_columns))]

        kept, sources = self._select_tables(top_columns, candidates, matches)
        # ``connect`` can introduce further bridge tables; they are part of the
        # answer, so they count towards the columns and the coverage guard.
        # ``sorted`` is load-bearing: ``connect`` breaks ties by input order, and
        # feeding it a set would make the join tree depend on PYTHONHASHSEED.
        ordered, edges = self.join_graph.connect(sorted(kept))
        final = set(ordered) | kept
        for table in sorted(final - kept):
            sources.setdefault(table, "join-bridge")
        tables = ordered or sorted(final)
        keep_cols = self._select_columns(final, top_columns, edges, matches)

        linked = LinkedSchema(
            tables=tables,
            columns=keep_cols,
            glossary=[m.entry for m in matches],
            join_paths=self._join_paths(tables),
            evidence=self._evidence(matches, top_columns, candidates, sources),
            coverage=round(len(final) / max(1, len(self.schema.tables)), 4),
        )
        log.info(
            "schema linked",
            tables=len(linked.tables),
            columns=len(linked.columns),
            glossary=len(matches),
            coverage=linked.coverage,
        )
        return linked

    # -- signals ------------------------------------------------------------ #

    def _bm25(self, text: str, tokens: list[str], index: _Index) -> np.ndarray:
        """Okapi BM25 over the same documents the dense index uses."""
        query = _term_frequencies(" ".join(tokens) if tokens else text)
        scores = np.zeros(len(index.docs), dtype=np.float32)
        for term in query:
            idf = index.idf.get(term)
            if idf is None:
                continue
            for row, freq in index.postings.get(term, ()):
                norm = 1.0 - BM25_B + BM25_B * (index.doc_lens[row] / index.avg_len)
                scores[row] += idf * (freq * (BM25_K1 + 1.0)) / (freq + BM25_K1 * norm)
        return scores

    def _glossary_boosts(self, matches: list[GlossaryMatch]) -> dict[str, float]:
        boosts: dict[str, float] = {}
        for m in matches:
            for ref in m.entry.columns:
                table, _, column = ref.partition(".")
                if column and self.schema.column(table, column):
                    boosts[ref] = max(boosts.get(ref, 0.0), m.score)
            for table in m.entry.tables:
                if self.schema.table(table):
                    scaled = m.score * GLOSSARY_TABLE_SHARE
                    boosts[table] = max(boosts.get(table, 0.0), scaled)
        return boosts

    def _value_boosts(
        self,
        nq: NormalizedQuestion,
        tokens: list[str],
        matches: list[GlossaryMatch],
        index: _Index,
    ) -> dict[str, float]:
        """Match question literals against profiled values and code labels.

        Glossary surfaces are treated as literals too: a business term that is
        *also* a stored code label (실효 = ``CTRT_STAT_CD '02'``) is the
        strongest single piece of evidence this system has.
        """
        literals = {v.strip().lower() for v in nq.value_candidates if v and v.strip()}
        literals |= {m.surface.lower() for m in matches}
        # Bare numbers are excluded from the token path: '20' extracted from
        # "20만원" would otherwise match the *code value* '20' of every CHNL_CD
        # -style column.  A genuinely literal number reaches us as a
        # value_candidate, where the NLU has already decided it is one.
        exact_only = {
            t.lower() for t in tokens if not t.isdigit() and len(t) >= 2
        } - literals
        if not literals and not exact_only:
            return {}

        boosts: dict[str, float] = {}
        for ref, (values, labels, categorical) in index.value_index.items():
            best = 0.0
            for lit in literals:
                if lit in labels or lit in values:
                    best = max(best, 1.0)
                elif len(lit) >= 2 and any(lit in label for label in labels):
                    best = max(best, 0.6)
                elif len(lit) >= 2 and categorical and any(lit in v for v in values):
                    best = max(best, 0.5)
            if best < 1.0:
                for tok in exact_only:
                    if tok in labels or (categorical and tok in values):
                        best = max(best, 0.9)
            if best:
                boosts[ref] = best
        return boosts

    # -- selection ---------------------------------------------------------- #

    def _select_tables(
        self,
        top_columns: list[LinkCandidate],
        candidates: dict[str, LinkCandidate],
        matches: list[GlossaryMatch],
    ) -> tuple[set[str], dict[str, str]]:
        """Seeds → join bridges → scored FK expansion → ``min_tables`` floor."""
        cfg = self.settings.retrieval
        # Two rankings: ``table_score`` counts only corroborated columns and
        # drives admission, while ``ranked`` also counts dense-only hits and is
        # consulted solely by the min_tables recall floor.
        table_score: dict[str, float] = {}
        loose_score: dict[str, float] = {}
        for cand in top_columns:
            loose_score[cand.table] = max(loose_score.get(cand.table, 0.0), cand.total)
            if _corroborated(cand):
                table_score[cand.table] = max(table_score.get(cand.table, 0.0), cand.total)
        for name in self.schema.tables:
            prior = TABLE_PRIOR * candidates[name].total if name in candidates else 0.0
            table_score[name] = table_score.get(name, 0.0) + prior
            loose_score[name] = loose_score.get(name, 0.0) + prior

        ranked = sorted(loose_score, key=lambda t: (-loose_score[t], t))
        sources: dict[str, str] = {}
        kept: set[str] = set()

        # A table is seeded only by a column that clearly stands out; the tail of
        # the top-k column list is noise and must not drag whole tables along.
        best_column = max((c.total for c in top_columns), default=0.0)
        seed_floor = SEED_RATIO * best_column
        for cand in top_columns:
            if cand.total < seed_floor or len(kept) >= int(cfg.top_k_tables):
                break
            # Corroboration rule: dense similarity *alone* may not seed a table.
            # On cryptic physical names a 256-bucket hashing embedder puts short
            # documents (``CNTN 상담내용``) near the top by collision, which is
            # the exact failure this hybrid retriever exists to avoid.  A real
            # hit almost always also fires BM25, the glossary or a value match;
            # a dense-only hit can still enter later through FK expansion.
            if not _corroborated(cand):
                continue
            if cand.table not in kept and self.schema.table(cand.table):
                kept.add(cand.table)
                sources[cand.table] = "column"
        for m in matches:
            if m.score < GLOSSARY_SEED_FLOOR:
                continue
            for table in m.entry.tables:
                if self.schema.table(table) and table not in kept:
                    kept.add(table)
                    sources[table] = "glossary"

        # Bridges are not optional: without TB_AGNT there is no path from a
        # contract to its branch, so they bypass every budget below.
        _, edges = self.join_graph.connect(sorted(kept))
        for edge in edges:
            for table in (edge.left_table, edge.right_table):
                if table not in kept:
                    kept.add(table)
                    sources[table] = "join-bridge"

        admit = FK_ADMIT_RATIO * best_column
        reachable = self.join_graph.expand(set(kept), int(cfg.fk_expand_hops)) - kept
        for table in sorted(reachable, key=lambda t: (-table_score.get(t, 0.0), t)):
            if len(kept) >= int(cfg.top_k_tables):
                break
            if table_score.get(table, 0.0) >= admit:
                kept.add(table)
                sources[table] = "fk-expand"

        # Recall floor.  FK-adjacent tables come first: a floor that picks the
        # globally best-scoring leftover can land on a table with no path to the
        # rest, which buys prompt tokens and no join.
        if len(kept) < int(cfg.min_tables):
            adjacent = self.join_graph.expand(set(kept), 1) - kept
            order = [t for t in ranked if t in adjacent] + [t for t in ranked if t not in adjacent]
            for table in order:
                if len(kept) >= int(cfg.min_tables):
                    break
                if table not in kept:
                    kept.add(table)
                    sources[table] = "min-tables"
        return kept, sources

    def _select_columns(
        self,
        kept: set[str],
        top_columns: list[LinkCandidate],
        edges: list[JoinEdge],
        matches: list[GlossaryMatch],
    ) -> list[str]:
        """Scored columns plus everything the SQL cannot be written without."""
        wanted: set[str] = {c.ref for c in top_columns if c.table in kept}
        for m in matches:
            for ref in m.entry.columns:
                table, _, column = ref.partition(".")
                if table in kept and column and self.schema.column(table, column):
                    wanted.add(ref)
        for name in sorted(kept):
            info = self.schema.table(name)
            if info:
                wanted.update(f"{info.name}.{pk}" for pk in info.primary_key)
        for edge in edges:
            for left, right in edge.on:
                wanted.add(f"{edge.left_table}.{left}")
                wanted.add(f"{edge.right_table}.{right}")

        # Emit in schema order so the rendered card is stable across runs.
        ordered: list[str] = []
        for info in self.schema.tables.values():
            if info.name not in kept:
                continue
            ordered.extend(c.qualified for c in info.columns if c.qualified in wanted)
        return ordered

    def _join_paths(self, tables: list[str]) -> list[list[ForeignKey]]:
        """Root-to-table FK chains derived from the Steiner tree in ``connect``."""
        if len(tables) < 2:
            return []
        root = tables[0]
        paths: list[list[ForeignKey]] = []
        seen: set[str] = set()
        for target in tables[1:]:
            edges = self.join_graph.shortest_path(root, target)
            if not edges:
                continue
            fks = [fk for edge in edges for fk in _edge_to_fks(edge)]
            key = "|".join(fk.key for fk in fks)
            if key in seen:
                continue
            seen.add(key)
            paths.append(fks)
        return paths

    def _evidence(
        self,
        matches: list[GlossaryMatch],
        top_columns: list[LinkCandidate],
        candidates: dict[str, LinkCandidate],
        sources: dict[str, str],
    ) -> list[ScoredItem]:
        items: list[ScoredItem] = [
            ScoredItem(ref=f"glossary:{m.entry.term}", score=round(m.score, 4), source="glossary")
            for m in matches
        ]
        for cand in top_columns[:MAX_EVIDENCE]:
            items.append(
                ScoredItem(ref=cand.ref, score=round(cand.total, 4), source=_dominant(cand, self.settings))
            )
        for table, why in sources.items():
            if why in {"join-bridge", "fk-expand", "min-tables"}:
                score = candidates[table].total if table in candidates else 0.0
                items.append(ScoredItem(ref=table, score=round(score, 4), source=why))
        items.sort(key=lambda s: (-s.score, s.ref))
        return items[:MAX_EVIDENCE]


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #


def _term_frequencies(text: str) -> dict[str, int]:
    """BM25 terms: identifier words, their underscore parts, Hangul bigrams."""
    tf: dict[str, int] = defaultdict(int)
    for word in _WORD_RE.findall(text.lower()):
        tf[word] += 1
        if "_" in word:
            for part in word.split("_"):
                if part:
                    tf[part] += 1
        elif "가" <= word[0] <= "힣":
            for i in range(len(word) - 1):
                tf[word[i : i + 2]] += 1
    return dict(tf)


def _normalise_signal(scores: np.ndarray) -> np.ndarray:
    """Rescale one signal to ``[0, 1]`` *relative to its own distribution*.

    Plain max-scaling is wrong here: a hashing embedder gives every document a
    non-zero similarity floor from n-gram collisions, so an irrelevant column
    lands at 0.3 and survives any fixed threshold.  Standardising first answers
    the question that actually matters — "how many standard deviations above the
    field is this document?" — which collapses the floor to zero and keeps the
    fusion weights meaningful across embedders with different score scales.
    """
    if scores.size == 0:
        return scores
    top = float(scores.max())
    if top <= 0.0:
        return np.zeros_like(scores)
    std = float(scores.std())
    if std <= 1e-9:  # degenerate: every document scored the same
        return scores / top
    z = np.maximum((scores - float(scores.mean())) / std, 0.0)
    peak = float(z.max())
    return z / peak if peak > 0 else z


def _is_indexable_category(profile: ColumnProfile) -> bool:
    """Whether a column's sampled values are safe to put into the retrieval index.

    ``is_categorical`` is a cardinality test, and cardinality alone is fooled by
    a canned-response column: 18 distinct complaint bodies look categorical and
    are anything but.  Word count is the sharper discriminator — a code label is
    one or two words (``표준체 승낙``), a complaint is a sentence.
    """
    if not (profile.is_categorical and profile.values):
        return False
    if profile.avg_length > MAX_INDEXED_VALUE_CHARS:
        return False
    words = sum(len(v.split()) for v in profile.values) / len(profile.values)
    return words <= MAX_INDEXED_VALUE_WORDS


def _corroborated(cand: LinkCandidate) -> bool:
    """True when something other than embedding similarity supports this column."""
    return (cand.lexical + cand.glossary + cand.value) > 0.0


def _edge_to_fks(edge: JoinEdge) -> list[ForeignKey]:
    return [ForeignKey(edge.left_table, lc, edge.right_table, rc) for lc, rc in edge.on]


def _dominant(cand: LinkCandidate, settings: Settings) -> str:
    cfg = settings.retrieval
    weighted = {
        "dense": float(cfg.dense_weight) * cand.dense,
        "lexical": (1.0 - float(cfg.dense_weight)) * cand.lexical,
        "glossary": float(cfg.glossary_weight) * cand.glossary,
        "value": float(cfg.value_match_weight) * cand.value,
    }
    return max(weighted, key=lambda k: weighted[k])
