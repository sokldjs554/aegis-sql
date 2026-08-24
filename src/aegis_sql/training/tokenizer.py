"""Byte-level BPE trained on *this* corpus, because off-the-shelf vocabularies lose to it here.

Every general-purpose tokenizer we could have shipped is wrong for the AEGIS
workload in the same direction: it was fitted on web text, and this corpus is
Korean insurance jargon plus SQL over ``UPPER_SNAKE_CASE`` physical names.  A
multilingual sentencepiece model spends 3-4 tokens on ``계약상태코드`` and shreds
``CTRT_STAT_CD`` into six pieces, so a 512-token context holds roughly half the
schema card it should.  Fitting BPE on the flywheel corpus instead lets frequent
domain strings (``TB_CTRT``, ``GROUP BY``, ``납입주기코드``) collapse into single
tokens, which is a direct, measurable increase in usable context — and context
is the whole game for a 15M-parameter model that cannot afford a truncated
schema.

Design choices worth knowing:

* **Byte level, not character level.**  The base alphabet is the 256 byte
  values, so the vocabulary is closed: every possible input encodes, and
  ``decode(encode(s)) == s`` holds exactly for Korean, for SQL, for emoji and
  for the mixed strings that dominate real questions.  There is no ``<unk>`` in
  this tokenizer, and no round-trip loss to debug at 3am.
* **Chunked training, not stream training.**  The corpus is pre-tokenized into
  word-ish chunks (a GPT-2 style regex extended so that ``_`` stays glued to
  identifiers) and collapsed into a ``chunk -> count`` table.  Merges then
  operate on a few tens of thousands of unique chunks instead of tens of
  millions of bytes.
* **Incremental pair statistics.**  Recounting every adjacent pair after each
  merge is O(corpus) per merge and turns an 8k-vocab fit into a coffee break.
  We keep a ``pair -> count`` table plus a ``pair -> chunk ids`` index and touch
  only the chunks a merge actually changed, with a lazy heap for the argmax.
* **Deterministic.**  Ties in merge frequency are broken by token id, so the
  same corpus always yields byte-identical vocabularies across machines and
  runs — a precondition for a checkpoint and its tokenizer staying in sync.
"""

from __future__ import annotations

import heapq
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from aegis_sql.observability.logging import get_logger

log = get_logger("training.tokenizer")

#: Default special tokens.  ``<|sql|>`` is the hard boundary between the prompt
#: and the SQL the model must produce; SFT masks the loss up to and including it.
PAD_TOKEN = "<|pad|>"
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
SEP_TOKEN = "<|sql|>"
DEFAULT_SPECIALS: tuple[str, ...] = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN)

TOKENIZER_FILE = "tokenizer.json"

#: Word-ish chunker.  Order matters: digits, then letters+underscore (so
#: ``CTRT_STAT_CD`` survives as one chunk and never merges across a space),
#: then punctuation runs, then whitespace.  Every codepoint matches exactly one
#: alternative, so ``"".join(findall(t)) == t`` for any input.
_PRETOKEN_RE = re.compile(r" ?\d+| ?[^\W\d]+| ?[^\s\w]+|\s+(?!\S)|\s+")

_Pair = tuple[int, int]


class ByteBPETokenizer:
    """A byte-level BPE tokenizer: 256 byte tokens, N specials, and learned merges.

    Ids are laid out ``[specials][256 bytes][merges...]`` so that ``pad_id == 0``
    and the byte block sits at a fixed, checkpoint-stable offset.
    """

    __slots__ = ("_cache", "_id_to_bytes", "_merges", "_ranks", "_specials", "_special_ids")

    def __init__(self, specials: list[str], merges: list[_Pair]) -> None:
        self._specials: list[str] = list(specials)
        self._special_ids: dict[str, int] = {tok: i for i, tok in enumerate(self._specials)}
        base = len(self._specials)

        # id -> the byte string it decodes to.  Specials decode to nothing.
        self._id_to_bytes: list[bytes] = [b""] * base + [bytes([b]) for b in range(256)]
        self._merges: list[_Pair] = list(merges)
        self._ranks: dict[_Pair, int] = {}
        for rank, (left, right) in enumerate(self._merges):
            new_id = base + 256 + rank
            self._ranks[(left, right)] = rank
            self._id_to_bytes.append(self._id_to_bytes[left] + self._id_to_bytes[right])
            assert len(self._id_to_bytes) == new_id + 1
        self._cache: dict[str, list[int]] = {}

    # -- construction ------------------------------------------------------ #

    @classmethod
    def train(
        cls,
        corpus: list[str],
        vocab_size: int,
        special_tokens: list[str] | None = None,
        min_frequency: int = 2,
        seed: int = 20260824,
        max_lines: int | None = None,
    ) -> ByteBPETokenizer:
        """Fit merges on ``corpus`` until the vocabulary reaches ``vocab_size``.

        ``seed`` only drives the deterministic down-sample applied when the
        corpus exceeds ``max_lines``; the merge search itself is fully
        deterministic (frequency, then token id).
        """
        specials = list(special_tokens) if special_tokens else list(DEFAULT_SPECIALS)
        for required in DEFAULT_SPECIALS:
            if required not in specials:
                specials.append(required)
        base = len(specials)
        n_merges = max(0, vocab_size - base - 256)

        lines = corpus
        if max_lines is not None and len(lines) > max_lines:
            lines = random.Random(seed).sample(lines, max_lines)

        chunk_counts: dict[str, int] = defaultdict(int)
        for line in lines:
            for chunk in _PRETOKEN_RE.findall(line):
                chunk_counts[chunk] += 1

        words: list[list[int]] = []
        counts: list[int] = []
        for chunk, count in chunk_counts.items():
            symbols = [base + b for b in chunk.encode("utf-8")]
            if len(symbols) >= 2:
                words.append(symbols)
                counts.append(count)

        merges = _learn_merges(words, counts, n_merges, min_frequency, base + 256)
        log.info(
            "byte-BPE trained",
            lines=len(lines),
            unique_chunks=len(chunk_counts),
            merges=len(merges),
            vocab_size=base + 256 + len(merges),
        )
        return cls(specials, merges)

    # -- properties -------------------------------------------------------- #

    @property
    def vocab_size(self) -> int:
        return len(self._id_to_bytes)

    @property
    def specials(self) -> list[str]:
        return list(self._specials)

    @property
    def pad_id(self) -> int:
        return self._special_ids[PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self._special_ids[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self._special_ids[EOS_TOKEN]

    @property
    def sep_id(self) -> int:
        """Id of ``<|sql|>`` — the prompt/target boundary used by SFT and DPO."""
        return self._special_ids[SEP_TOKEN]

    def special_id(self, token: str) -> int:
        return self._special_ids[token]

    # -- encode / decode --------------------------------------------------- #

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode ``text``; special tokens are never produced from the text itself."""
        ids: list[int] = [self.bos_id] if add_bos else []
        for chunk in _PRETOKEN_RE.findall(text):
            cached = self._cache.get(chunk)
            if cached is None:
                cached = self._encode_chunk(chunk)
                if len(self._cache) < 200_000:
                    self._cache[chunk] = cached
            ids.extend(cached)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def _encode_chunk(self, chunk: str) -> list[int]:
        base = len(self._specials)
        symbols = [base + b for b in chunk.encode("utf-8")]
        if len(symbols) < 2:
            return symbols
        # Repeatedly apply the lowest-rank (earliest learned) applicable merge.
        while True:
            best_rank = len(self._merges)
            best_at = -1
            for i in range(len(symbols) - 1):
                rank = self._ranks.get((symbols[i], symbols[i + 1]))
                if rank is not None and rank < best_rank:
                    best_rank, best_at = rank, i
            if best_at < 0:
                return symbols
            new_id = base + 256 + best_rank
            symbols[best_at : best_at + 2] = [new_id]
            if len(symbols) == 1:
                return symbols

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Inverse of :meth:`encode`.  Exact for any input that encode produced."""
        pieces: list[bytes] = []
        n_special = len(self._specials)
        for tid in ids:
            if 0 <= tid < n_special:
                if not skip_special_tokens:
                    pieces.append(self._specials[tid].encode("utf-8"))
                continue
            if 0 <= tid < len(self._id_to_bytes):
                pieces.append(self._id_to_bytes[tid])
        return b"".join(pieces).decode("utf-8", errors="replace")

    # -- persistence ------------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "aegis-byte-bpe/1",
            "specials": self._specials,
            "vocab_size": self.vocab_size,
            "merges": [[a, b] for a, b in self._merges],
        }

    def save(self, path: str | Path) -> Path:
        """Write the tokenizer.  A directory (or extension-less path) gets ``tokenizer.json``."""
        target = _resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")
        log.info("tokenizer saved", path=str(target), vocab_size=self.vocab_size)
        return target

    @classmethod
    def load(cls, path: str | Path) -> ByteBPETokenizer:
        target = _resolve_path(path)
        payload = json.loads(target.read_text(encoding="utf-8"))
        merges = [(int(a), int(b)) for a, b in payload["merges"]]
        return cls(list(payload["specials"]), merges)

    def __repr__(self) -> str:  # pragma: no cover - display helper
        return f"ByteBPETokenizer(vocab_size={self.vocab_size}, merges={len(self._merges)})"


# --------------------------------------------------------------------------- #
# Merge learning
# --------------------------------------------------------------------------- #


def _learn_merges(
    words: list[list[int]],
    counts: list[int],
    n_merges: int,
    min_frequency: int,
    first_new_id: int,
) -> list[_Pair]:
    """Greedy BPE with incremental pair statistics and a lazy argmax heap.

    ``pair_where`` maps a pair to the chunk indices that currently contain it, so
    a merge only rewrites the chunks it actually touches.  The heap holds stale
    entries by design; they are discarded on pop when the stored count no longer
    matches ``pair_counts``.
    """
    pair_counts: dict[_Pair, int] = defaultdict(int)
    pair_where: dict[_Pair, set[int]] = defaultdict(set)
    for idx, symbols in enumerate(words):
        weight = counts[idx]
        for pair in zip(symbols, symbols[1:], strict=False):
            pair_counts[pair] += weight
            pair_where[pair].add(idx)

    heap: list[tuple[int, _Pair]] = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)

    merges: list[_Pair] = []
    while len(merges) < n_merges:
        best: _Pair | None = None
        while heap:
            neg_count, pair = heapq.heappop(heap)
            if pair_counts.get(pair, 0) == -neg_count and -neg_count >= min_frequency:
                best = pair
                break
        if best is None:
            break

        new_id = first_new_id + len(merges)
        merges.append(best)
        touched: set[_Pair] = set()

        for idx in list(pair_where.get(best, ())):
            symbols = words[idx]
            weight = counts[idx]
            merged = _apply_merge(symbols, best, new_id)
            if merged is None:
                continue
            for pair in set(zip(symbols, symbols[1:], strict=False)):
                pair_where.get(pair, _EMPTY).discard(idx)
            for pair, delta in _pair_delta(symbols, weight).items():
                pair_counts[pair] -= delta
                touched.add(pair)
            words[idx] = merged
            for pair, delta in _pair_delta(merged, weight).items():
                pair_counts[pair] += delta
                pair_where[pair].add(idx)
                touched.add(pair)

        pair_counts.pop(best, None)
        pair_where.pop(best, None)
        for pair in touched:
            count = pair_counts.get(pair, 0)
            if count > 0:
                heapq.heappush(heap, (-count, pair))
            else:
                pair_counts.pop(pair, None)
                pair_where.pop(pair, None)

    return merges


_EMPTY: set[int] = set()


def _pair_delta(symbols: list[int], weight: int) -> dict[_Pair, int]:
    out: dict[_Pair, int] = defaultdict(int)
    for pair in zip(symbols, symbols[1:], strict=False):
        out[pair] += weight
    return out


def _apply_merge(symbols: list[int], pair: _Pair, new_id: int) -> list[int] | None:
    """Return ``symbols`` with every occurrence of ``pair`` replaced, or None if absent."""
    left, right = pair
    out: list[int] = []
    i = 0
    n = len(symbols)
    hit = False
    while i < n:
        if i < n - 1 and symbols[i] == left and symbols[i + 1] == right:
            out.append(new_id)
            i += 2
            hit = True
        else:
            out.append(symbols[i])
            i += 1
    return out if hit else None


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_dir() or not p.suffix:
        return p / TOKENIZER_FILE
    return p
