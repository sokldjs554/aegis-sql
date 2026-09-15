# Spider-KO external evaluation

## Why this exists

KorFin-Bench measures the insurance/finance problem AEGIS-SQL was built for, but it is an internal benchmark. A strong project should also show what happens when the model sees a database schema that was not designed together with the system.

Spider-KO is an external Korean translation of the Spider Text-to-SQL benchmark. Its validation split has **1,034** examples and follows Spider's cross-domain database split. This evaluation is therefore kept separate from the production insurance engine and asks:

> Can the same HF Text-to-SQL generation approach operate on Korean questions and previously unseen relational schemas?

This is not a replacement for KorFin-Bench. The two numbers measure different things and must never be merged into one headline score.

## Data contract

Dataset:

- source: `huggingface-KREW/spider-ko`
- split: `validation`
- model input: `question_ko`
- expected rows: `1,034`
- license: CC BY-SA 4.0

Database execution substrate:

- source repo: `HAL-9001/spider-databases`
- pinned revision: `4a01bbac6520cd35b216db9e1724e5e1ada60aa4`
- archive: `spider_data.zip`
- expected SHA-256: `00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b`
- archive provenance: re-host of the canonical Yale Spider distribution
- expected extracted layout: `spider_data/database/<db_id>/<db_id>.sqlite`

The database bootstrap refuses a checksum mismatch, rejects unsafe ZIP paths/symlinks, validates the expected SQLite layout, and records source revision plus archive SHA-256 in a manifest. Database files themselves are not committed to this repository.

The Spider-KO preparation script resolves its Hugging Face revision to an immutable commit SHA before export. The evaluator additionally hashes every SQLite database touched by the run.

## What is deliberately held separate

The external path does **not** use the insurance dictionary, insurance common-code supervision, insurance governance policy as an accuracy aid, KorFin schema-linking labels, or KorFin training examples.

It reuses only generic infrastructure: SQLite schema introspection, `SchemaCardBuilder`, read-only/time-bounded `SQLExecutor`, AEGIS execution-result comparison, and HF/PEFT runtime metadata conventions.

This makes the experiment useful without pretending it is the official Spider leaderboard evaluator. The report explicitly names the comparison as `AEGIS execution_match`.

## Reproducible preparation

The runner performs these steps automatically. They can also be run independently:

```bash
pip install -e ".[hf]"
python scripts/prepare_spider_databases.py --out data/external/spider-db
python scripts/prepare_spider_ko.py \
  --out data/external/spider-ko-validation.jsonl \
  --manifest data/external/spider-ko-validation.manifest.json
```

A custom already-prepared Spider database can still be supplied through `SPIDER_DB_ROOT`.

## Smoke run

A smoke run proves wiring only and is **never** portfolio evidence:

```bash
SMOKE=1 bash scripts/run_spider_ko_colab.sh
```

The runner downloads and verifies the pinned database archive automatically. The smoke path evaluates only 10 rows, so `portfolio_evidence_ready=false`.

## Full GPU run

A full run requires persistent storage. This is intentional: the previous Qwen adaptation experiment completed successfully but its raw JSON/ZIP was lost when the Colab runtime reset before download.

After mounting Google Drive in a GPU Colab runtime:

```bash
ARCHIVE_DIR=/content/drive/MyDrive/aegis-spider-ko \
bash scripts/run_spider_ko_colab.sh
```

That single command prepares the pinned Spider DB archive, exports the pinned Spider-KO validation split, runs all 1,034 rows, bundles row-level evidence, and copies the result plus dataset/database provenance manifests to persistent storage.

Default model: `Qwen/Qwen2.5-Coder-1.5B-Instruct` in NF4 4-bit inference. `MODEL` and `ADAPTER` may be overridden, but different runs must be reported separately.

## Evidence gate

A report is marked `portfolio_evidence_ready=true` only when all of these are true:

1. benchmark id is exactly `huggingface-KREW/spider-ko`
2. split is `validation`
3. question language is `ko`
4. all **1,034** validation examples were evaluated
5. all 1,034 row-level records are present
6. every row preserves `db_id`, Korean question, gold SQL, predicted SQL, and correctness

The report additionally keeps per-item latency and execution errors, dataset SHA-256, per-database SHA-256, source Git SHA, model/adapter identity, quantization, runtime versions, and GPU metadata. The result bundle also includes the dataset and database provenance manifests when the pinned bootstrap path is used.

## Interpretation rules

Do not claim the score as official Spider leaderboard accuracy. This runner uses the original per-database SQLite files but AEGIS's execution-result comparator rather than Spider's official evaluator. Its purpose is a reproducible **external generalisation check** inside this project.

If the result is poor, that is still useful evidence. It separates domain-specialised performance from cross-domain generalisation and gives a concrete basis for the next ablation: schema representation, model scale, adaptation data, or decoding strategy. A larger model should only be tested after the 1.5B external baseline is archived successfully.
