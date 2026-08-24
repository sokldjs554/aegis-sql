"""The flywheel end to end: schema in, a verified Korean Text-to-SQL corpus out.

``build()`` is the entry point the CLI, the Makefile and the training package all
call.  It wires the four stages that turn a database into supervision —

    sample SQL  →  back-translate to Korean  →  augment  →  verify & split

— and, just as importantly, writes a manifest next to the data.  A dataset
without a manifest is an unreproducible artefact: six weeks later nobody can say
which schema fingerprint it came from, what the difficulty mix actually was
after filtering, how many pairs the executor rejected, or whether two checkpoints
were trained on the same rows.  The manifest here records all of that, including
a structural cross-check (``difficulty_of`` over the emitted SQL) against the
sampler's own curriculum labels, so a drift between "what we asked for" and "what
we got" shows up as a number rather than as a surprise.

Ordering note: augmentation runs *before* verification, not after.  Paraphrases
share their parent's SQL, so the executor is hit once per distinct statement
(``QualityFilter`` memoises), and the near-duplicate detector gets to compare a
paraphrase against its own parent — which is exactly the comparison that decides
whether the paraphrase was worth generating.

The output layout is the one ``training/`` consumes:

    {output_dir}/train.jsonl   {"question","sql","difficulty","template_id","source","schema","tables"}
    {output_dir}/dev.jsonl
    {output_dir}/test.jsonl
    {output_dir}/manifest.json
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from aegis_sql.config import PROJECT_ROOT, Settings
from aegis_sql.flywheel.augment import KoreanAugmenter
from aegis_sql.flywheel.back_translate import BackTranslator
from aegis_sql.flywheel.quality_filter import SPLITS, Pair, QualityFilter
from aegis_sql.flywheel.sql_sampler import SQLProgram, SQLSampler, template_ids
from aegis_sql.llm.base import LLMClient
from aegis_sql.observability.logging import get_logger
from aegis_sql.observability.trace import Tracer
from aegis_sql.schema.graph import JoinGraph
from aegis_sql.schema.introspect import introspect
from aegis_sql.schema.profile import Profiler
from aegis_sql.verify.executor import SQLExecutor

log = get_logger("flywheel.build_dataset")

__all__ = ["build", "load_split", "RECORD_FIELDS"]

#: JSONL schema.  Fixed and ordered so a diff between two builds is readable.
RECORD_FIELDS: tuple[str, ...] = (
    "question",
    "sql",
    "difficulty",
    "template_id",
    "source",
    "schema",
    "tables",
)


def build(
    settings: Settings,
    n_programs: int | None = None,
    augment_per_example: int | None = None,
    llm: LLMClient | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Build the corpus and return the manifest that was written beside it."""
    started = time.perf_counter()
    cfg = settings.flywheel
    n_programs = int(n_programs if n_programs is not None else cfg.n_programs)
    augment_per_example = int(
        augment_per_example if augment_per_example is not None else cfg.augment_per_example
    )
    output_dir = _resolve(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = _resolve(settings.database.path)
    tracer = Tracer("flywheel.build")

    with tracer.span("schema", db=db_path.name):
        schema = introspect(db_path)
        profile = Profiler(db_path, sample=settings.database.profile_sample).profile(
            schema, cache_path=output_dir / "schema_profile.json"
        )
        join_graph = JoinGraph(schema)
        glossary = _load_glossary(db_path)

    with tracer.span("sample", n=n_programs):
        sampler = SQLSampler(schema, profile, join_graph, glossary, seed=cfg.seed)
        programs = sampler.sample(n_programs, cfg.difficulty_mix)
    _report(progress, "sampled", len(programs), n_programs)

    with tracer.span("back_translate", n=len(programs)):
        translator = BackTranslator(
            schema, profile, glossary, llm=llm, registry=_registry(settings, llm), seed=cfg.seed
        )
        questions = translator.translate_batch(programs)
    _report(progress, "back-translated", len(questions), len(programs))

    with tracer.span("augment", per_example=augment_per_example):
        augmenter = KoreanAugmenter(glossary, seed=cfg.seed)
        pairs = _expand(programs, questions, augmenter, augment_per_example, schema.name)
    _report(progress, "augmented", len(pairs), len(programs) * (1 + augment_per_example))

    with tracer.span("filter", n=len(pairs)):
        executor = SQLExecutor(db_path, timeout_s=settings.database.timeout_s, max_rows=settings.database.max_rows)
        try:
            kept, stats = QualityFilter(executor, settings).filter(pairs)
        finally:
            executor.close()
    _report(progress, "verified", len(kept), len(pairs))

    manifest = _write(output_dir, kept, stats, schema, programs, settings, n_programs, augment_per_example)
    manifest["wall_time_s"] = round(time.perf_counter() - started, 2)
    manifest["trace"] = tracer.finish().to_dict()
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info(
        "dataset built",
        output=str(output_dir),
        pairs=manifest["counts"]["total"],
        **manifest["counts"]["splits"],
        seconds=manifest["wall_time_s"],
    )
    return manifest


def load_split(path: str | Path) -> list[Pair]:
    """Read one ``*.jsonl`` split back into :class:`Pair` objects."""
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"split not found: {file}")
    out: list[Pair] = []
    for line in file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        out.append(
            Pair(
                question=record["question"],
                sql=record["sql"],
                difficulty=record.get("difficulty", "medium"),
                template_id=record.get("template_id", ""),
                source=record.get("source", "backtranslate"),
                meta={
                    "schema": record.get("schema", ""),
                    "tables": record.get("tables", []),
                    "split": file.stem,
                },
            )
        )
    log.debug("split loaded", path=file.name, pairs=len(out))
    return out


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #


def _expand(
    programs: list[SQLProgram],
    questions: list[str],
    augmenter: KoreanAugmenter,
    per_example: int,
    schema_name: str,
) -> list[Pair]:
    """One back-translated pair per program, plus its augmented siblings."""
    pairs: list[Pair] = []
    for program, question in zip(programs, questions):
        base = {
            "schema": schema_name,
            "tables": list(program.tables),
            "slots": program.slots,
        }
        pairs.append(
            Pair(
                question=question,
                sql=program.sql,
                difficulty=program.difficulty,
                template_id=program.template_id,
                source="backtranslate",
                meta=dict(base),
            )
        )
        for variant, operations in augmenter.augment_with_ops(question, per_example):
            pairs.append(
                Pair(
                    question=variant,
                    sql=program.sql,
                    difficulty=program.difficulty,
                    template_id=program.template_id,
                    source="augment:" + "+".join(operations),
                    meta={**base, "parent": question, "transforms": operations},
                )
            )
    return pairs


def _write(
    output_dir: Path,
    kept: list[Pair],
    stats: dict[str, Any],
    schema: Any,
    programs: list[SQLProgram],
    settings: Settings,
    n_programs: int,
    augment_per_example: int,
) -> dict[str, Any]:
    """Write the three splits and assemble the manifest."""
    by_split: dict[str, list[Pair]] = {s: [] for s in SPLITS}
    for pair in kept:
        by_split.setdefault(pair.split or "train", []).append(pair)

    for split, members in by_split.items():
        lines = [json.dumps(_record(p), ensure_ascii=False) for p in members]
        (output_dir / f"{split}.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    return {
        "schema": {
            "name": schema.name,
            "fingerprint": schema.fingerprint(),
            "tables": len(schema.tables),
            "database": Path(settings.database.path).name,
        },
        "config": {
            "seed": settings.flywheel.seed,
            "n_programs_requested": n_programs,
            "n_programs_sampled": len(programs),
            "augment_per_example": augment_per_example,
            "difficulty_mix": dict(settings.flywheel.difficulty_mix),
            "require_nonempty_result": settings.flywheel.require_nonempty_result,
            "dedupe_threshold": settings.flywheel.dedupe_threshold,
            "templates_registered": len(template_ids()),
        },
        "counts": {
            "total": len(kept),
            "splits": {s: len(by_split.get(s, [])) for s in SPLITS},
            "by_difficulty": _counts(kept, lambda p: p.difficulty),
            "by_template": _counts(kept, lambda p: p.template_id),
            "by_source": _counts(kept, lambda p: p.source.split(":")[0]),
            "by_split_difficulty": {
                s: _counts(by_split.get(s, []), lambda p: p.difficulty) for s in SPLITS
            },
        },
        "filter": stats,
        # The sampler's buckets are a curriculum judgement; ``difficulty_of`` is a
        # structural one.  Publishing both keeps the gap between them honest.
        "difficulty_cross_check": _structural_difficulty(kept),
        "leakage": _leakage_report(by_split),
    }


def _record(pair: Pair) -> dict[str, Any]:
    return {
        "question": pair.question,
        "sql": pair.sql,
        "difficulty": pair.difficulty,
        "template_id": pair.template_id,
        "source": pair.source,
        "schema": pair.meta.get("schema", ""),
        "tables": pair.meta.get("tables", []),
    }


def _structural_difficulty(pairs: list[Pair]) -> dict[str, Any]:
    try:
        from aegis_sql.generation.skeleton import difficulty_of
    except Exception:  # pragma: no cover - generation/ not present
        return {"available": False}
    graded = Counter(difficulty_of(p.sql) for p in pairs)
    agreement = sum(1 for p in pairs if difficulty_of(p.sql) == p.difficulty)
    return {
        "available": True,
        "by_difficulty": dict(graded),
        "agreement_with_curriculum": round(agreement / len(pairs), 4) if pairs else 0.0,
    }


def _leakage_report(by_split: dict[str, list[Pair]]) -> dict[str, Any]:
    """Prove the property the split stage claims, in the artefact itself."""
    skeletons = {
        split: {str(p.meta.get("sql_skeleton", "")) for p in members}
        for split, members in by_split.items()
    }
    train, dev, test = (skeletons.get(s, set()) for s in SPLITS)
    return {
        "skeletons": {s: len(skeletons.get(s, set())) for s in SPLITS},
        "train_test_overlap": len(train & test),
        "train_dev_overlap": len(train & dev),
        "dev_test_overlap": len(dev & test),
    }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _counts(pairs: list[Pair], key: Any) -> dict[str, int]:
    return dict(sorted(Counter(key(p) for p in pairs).items()))


def _resolve(path: str | Path) -> Path:
    """``configs/default.yaml`` stores repo-relative paths; make them absolute."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)


def _load_glossary(db_path: Path) -> Any:
    """Load the 용어사전 that ships next to the database, if there is one."""
    try:
        from aegis_sql.retrieval.glossary import Glossary
    except Exception:  # pragma: no cover - retrieval/ not present
        return None
    path = db_path.parent / "glossary.yaml"
    return Glossary.load(path) if path.exists() else None


def _registry(settings: Settings, llm: LLMClient | None) -> Any:
    """The prompt set, loaded only when an LLM rewriter will actually use it."""
    if llm is None:
        return None
    try:
        from aegis_sql.prompts.registry import PromptRegistry

        return PromptRegistry.load(settings.generation.prompt_set)
    except Exception as exc:  # pragma: no cover - missing prompt set is not fatal
        log.warning("prompt set unavailable; llm rewriting disabled", error=str(exc))
        return None


def _report(progress: bool, stage: str, produced: int, expected: int) -> None:
    if not progress:
        return
    log.info(f"flywheel: {stage}", produced=produced, expected=expected)
