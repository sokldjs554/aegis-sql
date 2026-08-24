#!/usr/bin/env python3
"""Train the in-house sLLM end to end: tokenizer → SFT → (LoRA) → (DPO) → checkpoint.

    python scripts/train_slm.py --data-dir data/generated/flywheel --out data/generated/slm/aegis-lm-tiny
    python scripts/train_slm.py --epochs 2 --lora --dpo          # fresh checkout, no corpus needed

Input is the flywheel's JSONL corpus — ``{"question": ..., "sql": ..., "schema_card": ...}`` rows,
optionally split across ``train.jsonl`` / ``dev.jsonl``.  Preference pairs for the optional DPO pass
come from the repair log (``{"prompt", "gold_sql", "failed_sql"}``), which is the engine's own record
of what it got wrong in production.

If no corpus is present the script *synthesises* one from the demo database — schema-grounded
templates whose literals are sampled from real column values, and every statement is executed before
it enters the corpus so nothing unrunnable is trained on.  That keeps ``make train-slm`` working on a
fresh checkout, and it says so loudly, because a model fitted on its own generator has learned the
generator, not the domain.

The whole run is CPU-only and seeded: same corpus + same config ⇒ same checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aegis_sql.config import Settings, TrainingConfig  # noqa: E402
from aegis_sql.observability.logging import configure_logging, get_logger  # noqa: E402
from aegis_sql.schema.card import SchemaCardBuilder  # noqa: E402
from aegis_sql.schema.introspect import introspect  # noqa: E402
from aegis_sql.training.dpo import DPOExample, DPOTrainer, pairs_from_repair_log  # noqa: E402
from aegis_sql.training.lora import (  # noqa: E402
    apply_lora,
    mark_only_lora_trainable,
    merge_lora_,
    save_lora,
)
from aegis_sql.training.model import AegisLM, AegisLMConfig  # noqa: E402
from aegis_sql.training.sft import SFTExample, SFTTrainer, build_prompt, seed_everything  # noqa: E402
from aegis_sql.training.tokenizer import TOKENIZER_FILE, ByteBPETokenizer  # noqa: E402
from aegis_sql.types import LinkedSchema, SchemaGraph  # noqa: E402

log = get_logger("scripts.train_slm")

SYNTHETIC_BANNER = """
============================================================================
  !!  합성 코퍼스로 학습합니다 (SYNTHETIC TRAINING CORPUS)  !!
  플라이휠 데이터셋({path})이 없어 데모 DB에서 직접
  스키마 기반 템플릿으로 {n}쌍을 만들었습니다. 모든 SQL은 실행 검증을
  거쳤지만, 질문 표현의 다양성은 실제 사용자 로그에 크게 못 미칩니다.
  배포 전 반드시 `aegis flywheel build` 산출물로 재학습하세요.
============================================================================
"""

PREFERENCE_FILES = ("preferences.jsonl", "repair_log.jsonl", "dpo.jsonl")
QUESTION_KEYS = ("question", "nlq", "nl", "utterance")
SQL_KEYS = ("sql", "gold_sql", "query", "target")


# --------------------------------------------------------------------------- #
# Corpus loading
# --------------------------------------------------------------------------- #


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _first(record: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def records_to_examples(
    records: list[dict[str, Any]], card_for: Any
) -> list[SFTExample]:
    """Map corpus rows onto rendered SFT pairs, skipping rows without both halves."""
    out: list[SFTExample] = []
    for record in records:
        question = _first(record, QUESTION_KEYS)
        sql = _first(record, SQL_KEYS)
        if not question or not sql:
            continue
        card = record.get("schema_card") or record.get("card") or ""
        if not card:
            card = card_for(record.get("tables") or [])
        out.append(SFTExample(prompt=build_prompt(question, str(card)), target=sql))
    return out


def load_corpus(
    data_dir: Path, card_for: Any
) -> tuple[list[SFTExample], list[SFTExample], list[dict[str, Any]]]:
    """Return ``(train, dev, preference_records)`` from a flywheel output directory."""
    if not data_dir.exists():
        return [], [], []

    preference: list[dict[str, Any]] = []
    for name in PREFERENCE_FILES:
        path = data_dir / name
        if path.exists():
            preference.extend(read_jsonl(path))

    train_rows = read_jsonl(data_dir / "train.jsonl") if (data_dir / "train.jsonl").exists() else []
    dev_rows = read_jsonl(data_dir / "dev.jsonl") if (data_dir / "dev.jsonl").exists() else []
    if not train_rows:
        skip = set(PREFERENCE_FILES) | {"dev.jsonl"}
        for path in sorted(data_dir.glob("*.jsonl")):
            if path.name not in skip:
                train_rows.extend(read_jsonl(path))
    # Honour an explicit split column when the corpus carries one.
    if not dev_rows:
        dev_rows = [r for r in train_rows if str(r.get("split", "")).lower() in {"dev", "valid", "val"}]
        train_rows = [r for r in train_rows if str(r.get("split", "")).lower() not in {"dev", "valid", "val"}]

    return records_to_examples(train_rows, card_for), records_to_examples(dev_rows, card_for), preference


# --------------------------------------------------------------------------- #
# Synthetic fallback corpus (schema-grounded, execution-verified)
# --------------------------------------------------------------------------- #

_AGG_KO = {"SUM": "합계", "AVG": "평균", "MAX": "최대", "MIN": "최소"}


def synthesise_corpus(
    db_path: Path, schema: SchemaGraph, builder: SchemaCardBuilder, n: int, seed: int
) -> list[SFTExample]:
    """Generate execution-verified (question, SQL) pairs straight from the database.

    Literals are sampled from the column's real values, so no template ever produces a
    predicate the data cannot satisfy.  Statements that fail to execute are discarded.
    """
    rng = random.Random(seed)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    examples: list[SFTExample] = []
    seen: set[str] = set()

    try:
        code_labels = _load_code_labels(conn)
        tables = [t for t in schema.tables.values() if t.name.upper() != "TB_COMM_CD"]
        attempts = 0
        while len(examples) < n and attempts < n * 12:
            attempts += 1
            table = rng.choice(tables)
            card = builder.render(LinkedSchema(tables=[table.name]), style="slm")
            label = table.comment or table.name
            columns = table.columns

            categorical = [c for c in columns if c.name.endswith("_CD") or c.name.endswith("_YN")]
            numeric = [c for c in columns if c.dtype.upper() in {"INTEGER", "REAL", "NUMERIC"}
                       and not c.is_primary_key and not c.name.endswith("_CD")]
            dates = [c for c in columns if c.name.endswith("_DT")]

            choice = rng.random()
            if choice < 0.22 or not categorical:
                question = f"{label} 전체 건수를 알려줘"
                sql = f"SELECT COUNT(*) FROM {table.name}"
            elif choice < 0.5:
                col = rng.choice(categorical)
                value = _sample_value(conn, table.name, col.name, rng)
                if value is None:
                    continue
                value_label = code_labels.get((col.code_group or "", value), value)
                question = f"{col.label}가 {value_label}인 {label} 건수는?"
                sql = f"SELECT COUNT(*) FROM {table.name} WHERE {col.name} = '{value}'"
            elif choice < 0.72 and numeric:
                col, group = rng.choice(numeric), rng.choice(categorical)
                agg = rng.choice(list(_AGG_KO))
                question = f"{group.label}별 {col.label} {_AGG_KO[agg]}를 구해줘"
                sql = (
                    f"SELECT {group.name}, {agg}({col.name}) AS metric FROM {table.name} "
                    f"GROUP BY {group.name} ORDER BY metric DESC"
                )
            elif choice < 0.88 and dates:
                col = rng.choice(dates)
                bounds = _date_bounds(conn, table.name, col.name)
                if not bounds:
                    continue
                lo, hi = bounds
                question = f"{lo[:4]}년 {label} 건수를 {col.label} 기준으로 알려줘"
                sql = (
                    f"SELECT COUNT(*) FROM {table.name} "
                    f"WHERE {col.name} BETWEEN '{lo}' AND '{hi}'"
                )
            else:
                group = rng.choice(categorical)
                question = f"{label}를 {group.label} 기준으로 집계해줘"
                sql = (
                    f"SELECT {group.name}, COUNT(*) AS cnt FROM {table.name} "
                    f"GROUP BY {group.name} ORDER BY cnt DESC"
                )

            key = " ".join(sql.lower().split())
            if key in seen or not _executes(conn, sql):
                continue
            seen.add(key)
            examples.append(SFTExample(prompt=build_prompt(question, card), target=sql))
    finally:
        conn.close()
    return examples


def _load_code_labels(conn: sqlite3.Connection) -> dict[tuple[str, str], str]:
    try:
        rows = conn.execute("SELECT CD_GRP, CD, CD_NM FROM TB_COMM_CD").fetchall()
    except sqlite3.Error:
        return {}
    return {(str(grp), str(code)): str(name) for grp, code, name in rows}


def _sample_value(conn: sqlite3.Connection, table: str, column: str, rng: random.Random) -> str | None:
    try:
        rows = conn.execute(
            f"SELECT {column} FROM {table} WHERE {column} IS NOT NULL "
            f"GROUP BY {column} ORDER BY COUNT(*) DESC LIMIT 12"
        ).fetchall()
    except sqlite3.Error:
        return None
    values = [str(r[0]) for r in rows if str(r[0]).strip() and "'" not in str(r[0])]
    return rng.choice(values) if values else None


def _date_bounds(conn: sqlite3.Connection, table: str, column: str) -> tuple[str, str] | None:
    try:
        lo, hi = conn.execute(f"SELECT MIN({column}), MAX({column}) FROM {table}").fetchone()
    except sqlite3.Error:
        return None
    if not lo or not hi or len(str(lo)) != 8:
        return None
    year = str(hi)[:4]
    return f"{year}0101", f"{year}1231"


def _executes(conn: sqlite3.Connection, sql: str) -> bool:
    try:
        conn.execute(f"SELECT * FROM ({sql}) LIMIT 1").fetchall()
        return True
    except sqlite3.Error:
        return False


def synthesise_preferences(examples: list[SFTExample], n: int, seed: int) -> list[DPOExample]:
    """Corrupt gold SQL into plausible near-misses — the mistakes a small model actually makes."""
    rng = random.Random(seed)
    records: list[dict[str, str]] = []
    for example in examples:
        if len(records) >= n:
            break
        gold = example.target
        broken = gold
        if " WHERE " in gold:  # drop the filter — the single most common sLLM failure
            broken = gold.split(" WHERE ")[0]
        elif " GROUP BY " in gold:
            broken = gold.split(" GROUP BY ")[0]
        elif "COUNT(*)" in gold:
            broken = gold.replace("COUNT(*)", "*", 1)
        if " ".join(broken.lower().split()) == " ".join(gold.lower().split()):
            continue
        records.append({"prompt": example.prompt, "gold_sql": gold, "failed_sql": broken})
    rng.shuffle(records)
    return pairs_from_repair_log(records)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:
    settings = Settings()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", default=settings.flywheel.output_dir, help="flywheel corpus directory")
    parser.add_argument("--out", default=settings.generation.slm_checkpoint, help="checkpoint directory")
    parser.add_argument("--db", default=settings.database.path, help="database used for the fallback corpus")
    parser.add_argument("--epochs", type=int, default=settings.training.epochs)
    parser.add_argument("--batch-size", type=int, default=settings.training.batch_size)
    parser.add_argument("--lr", type=float, default=settings.training.lr)
    parser.add_argument("--max-seq-len", type=int, default=settings.training.max_seq_len)
    parser.add_argument("--vocab-size", type=int, default=settings.training.vocab_size)
    parser.add_argument("--d-model", type=int, default=settings.training.d_model)
    parser.add_argument("--n-layers", type=int, default=settings.training.n_layers)
    parser.add_argument("--n-heads", type=int, default=settings.training.n_heads)
    parser.add_argument("--d-ff", type=int, default=settings.training.d_ff)
    parser.add_argument("--limit", type=int, default=0, help="cap the corpus (0 = no cap)")
    parser.add_argument("--dev-frac", type=float, default=0.05)
    parser.add_argument("--synth-size", type=int, default=1500, help="pairs to synthesise with no corpus")
    parser.add_argument("--lora", action=argparse.BooleanOptionalAction, default=False,
                        help="train LoRA adapters instead of the full model")
    parser.add_argument("--dpo", action="store_true", help="run a DPO pass after SFT")
    parser.add_argument("--dpo-steps", type=int, default=0, help="cap DPO steps (0 = full pass)")
    parser.add_argument(
        "--select-by", default="loss", choices=["loss", "token-acc"],
        help="dev metric that picks the checkpoint — the two disagree here, see docs/SLM.md §6",
    )
    parser.add_argument("--device", default=settings.training.device)
    parser.add_argument("--seed", type=int, default=settings.training.seed)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(args.log_level)
    started = time.perf_counter()
    seed_everything(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    schema = introspect(args.db)
    builder = SchemaCardBuilder(schema)
    full_card = builder.render(None, style="slm")

    def card_for(tables: Any) -> str:
        names = [t for t in (tables or []) if schema.table(str(t))]
        return builder.render(LinkedSchema(tables=names), style="slm") if names else full_card

    # -- corpus --------------------------------------------------------------- #
    train_examples, dev_examples, preference_records = load_corpus(data_dir, card_for)
    synthetic = not train_examples
    if synthetic:
        train_examples = synthesise_corpus(
            Path(args.db), schema, builder, args.synth_size, args.seed
        )
        print(SYNTHETIC_BANNER.format(path=data_dir, n=len(train_examples)), file=sys.stderr)
    if not train_examples:
        log.error("no training data", data_dir=str(data_dir))
        return 1

    rng = random.Random(args.seed)
    rng.shuffle(train_examples)
    if args.limit:
        train_examples = train_examples[: args.limit]
    if not dev_examples:
        n_dev = max(1, min(len(train_examples) // 10, int(len(train_examples) * args.dev_frac)))
        dev_examples, train_examples = train_examples[:n_dev], train_examples[n_dev:]

    # -- tokenizer ------------------------------------------------------------ #
    tokenizer_path = out_dir / TOKENIZER_FILE
    if tokenizer_path.exists():
        tokenizer = ByteBPETokenizer.load(tokenizer_path)
        log.info("tokenizer reused", path=str(tokenizer_path), vocab_size=tokenizer.vocab_size)
    else:
        corpus = [e.prompt for e in train_examples] + [e.target for e in train_examples] + [full_card]
        tokenizer = ByteBPETokenizer.train(corpus, vocab_size=args.vocab_size, special_tokens=None,
                                           min_frequency=2, seed=args.seed)
        tokenizer.save(tokenizer_path)

    # -- model ---------------------------------------------------------------- #
    cfg = TrainingConfig(
        output_dir=str(out_dir), d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        d_ff=args.d_ff, max_seq_len=args.max_seq_len, vocab_size=tokenizer.vocab_size,
        batch_size=args.batch_size, lr=args.lr, epochs=args.epochs, seed=args.seed, device=args.device,
    )
    model_cfg = AegisLMConfig(
        vocab_size=tokenizer.vocab_size, d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads,
        d_ff=cfg.d_ff, max_seq_len=cfg.max_seq_len, dropout=cfg.dropout,
        meta={"corpus": "synthetic" if synthetic else str(data_dir), "seed": args.seed},
    )
    model = AegisLM(model_cfg)
    total_params = model.num_parameters()

    lora_info: dict[str, Any] | None = None
    if args.lora:
        wrapped = apply_lora(model, cfg.lora_targets, r=cfg.lora_rank, alpha=cfg.lora_alpha,
                             dropout=cfg.lora_dropout)
        trainable, total = mark_only_lora_trainable(model)
        lora_info = {"modules": wrapped, "targets": cfg.lora_targets, "r": cfg.lora_rank,
                     "alpha": cfg.lora_alpha, "trainable": trainable, "total": total,
                     "trainable_pct": round(100.0 * trainable / max(1, total), 3)}

    # -- SFT ------------------------------------------------------------------ #
    trainer = SFTTrainer(model, tokenizer, cfg, device=args.device, select_by=args.select_by)
    sft_history = trainer.train(train_examples, dev_examples, output_dir=out_dir)

    if args.lora:
        save_lora(model, out_dir)
        merge_lora_(model)  # ship a plain checkpoint; the adapter stays on disk beside it
    model.save_pretrained(out_dir)
    tokenizer.save(out_dir)

    # -- DPO ------------------------------------------------------------------ #
    dpo_history: dict[str, Any] | None = None
    if args.dpo:
        pairs = pairs_from_repair_log(preference_records)
        synthetic_pairs = not pairs
        if synthetic_pairs:
            pairs = synthesise_preferences(train_examples, max(32, len(train_examples) // 10), args.seed)
        if pairs:
            reference = copy.deepcopy(model)
            dpo = DPOTrainer(model, reference, tokenizer, cfg, beta=cfg.dpo_beta, device=args.device)
            dpo_history = dpo.train(pairs, output_dir=out_dir,
                                    max_steps=args.dpo_steps or None)
            dpo_history["pairs"] = len(pairs)
            dpo_history["synthetic_pairs"] = synthetic_pairs
        else:
            log.warning("DPO requested but no preference pairs could be built")

    # -- report --------------------------------------------------------------- #
    report = {
        "checkpoint": str(out_dir),
        "corpus": {
            "source": "synthetic" if synthetic else str(data_dir),
            "train": len(train_examples),
            "dev": len(dev_examples),
            "preference_records": len(preference_records),
        },
        "tokenizer": {"vocab_size": tokenizer.vocab_size, "file": TOKENIZER_FILE},
        "model": {
            "params": total_params,
            **{k: getattr(model_cfg, k) for k in ("d_model", "n_layers", "n_heads", "d_ff", "max_seq_len")},
        },
        "lora": lora_info,
        "sft": sft_history,
        "dpo": dpo_history,
        "wall_s": round(time.perf_counter() - started, 2),
        "seed": args.seed,
    }
    (out_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print_summary(report)
    return 0


def print_summary(report: dict[str, Any]) -> None:
    corpus, model, sft = report["corpus"], report["model"], report["sft"]
    line = "-" * 74
    print(f"\n{line}\n  AEGIS-SQL sLLM training report\n{line}")
    print(f"  corpus      : {corpus['train']:,} train / {corpus['dev']:,} dev   (source: {corpus['source']})")
    print(f"  tokenizer   : byte-BPE, vocab {report['tokenizer']['vocab_size']:,}")
    print(f"  model       : {model['params']:,} params  ({model['d_model']}d x "
          f"{model['n_layers']}L x {model['n_heads']}H, ctx {model['max_seq_len']})")
    if report["lora"]:
        lora = report["lora"]
        print(f"  LoRA        : {lora['modules']} modules on {lora['targets']} "
              f"(r={lora['r']}) -> {lora['trainable']:,} trainable ({lora['trainable_pct']}%)")
    print(f"  SFT         : {sft['steps']} steps in {sft['wall_s']}s   "
          f"train loss {sft['train_loss'][0]:.3f} -> {sft['train_loss'][-1]:.3f}")
    if sft["dev_loss"]:
        print(f"                dev loss {sft['dev_loss'][0]:.3f} -> {sft['dev_loss'][-1]:.3f}   "
              f"token acc {sft['dev_token_acc'][-1]:.3f}")
    if report["dpo"]:
        dpo = report["dpo"]
        tag = " (synthetic pairs)" if dpo.get("synthetic_pairs") else ""
        print(f"  DPO         : {dpo['steps']} steps on {dpo['pairs']} pairs{tag}   "
              f"margin {dpo['margin_start']:.3f} -> {dpo['margin_end']:.3f}, acc {dpo['final_acc']:.2f}")
    print(f"  checkpoint  : {report['checkpoint']}")
    print(f"  total       : {report['wall_s']}s\n{line}")
    print("  다음 단계: aegis ask --tier slm \"정상 계약 건수는?\"\n")


if __name__ == "__main__":
    raise SystemExit(main())
