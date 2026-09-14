# Spider-KO external evaluation

## Why this exists

KorFin-Bench measures the insurance/finance problem AEGIS-SQL was built for, but it is an internal benchmark. A strong project should also show what happens when the model sees a database schema that was not designed together with the system.

Spider-KO is an external Korean translation of the Spider Text-to-SQL benchmark. Its validation split has **1,034** examples and follows Spider's cross-domain database split. This evaluation is therefore kept separate from the production insurance engine and answers a different question:

> Can the same HF Text-to-SQL generation approach operate on Korean questions and previously unseen relational schemas?

This is not a replacement for KorFin-Bench. The two numbers measure different things and must never be merged into one headline score.

## Data contract

Dataset:

- source: `huggingface-KREW/spider-ko`
- split: `validation`
- language used as model input: `question_ko`
- expected rows: `1,034`
- license: CC BY-SA 4.0 (follow the dataset card and original Spider attribution requirements)

Database files:

- use the SQLite databases distributed with the original Spider data
- expected layout: `<SPIDER_DB_ROOT>/<db_id>/<db_id>.sqlite`
- database files are not committed to this repository

The preparation script exports the selected external split to JSONL and records its SHA-256. The evaluator also hashes every SQLite database touched by the run.

## What is deliberately held separate

The external path does **not** use:

- the insurance dictionary
- insurance common-code inference as supervision
- the insurance governance policy as an accuracy aid
- KorFin schema-linking labels
- KorFin training examples

It does reuse generic, non-domain-specific infrastructure:

- SQLite schema introspection
- `SchemaCardBuilder`
- read-only/time-bounded `SQLExecutor`
- AEGIS execution-result comparison
- HF/PEFT model loading and runtime metadata conventions

This makes the experiment useful without pretending it is the official Spider leaderboard evaluator. The report explicitly names the comparison as `AEGIS execution_match`.

## Reproducible preparation

Install the research stack and export the fixed Korean validation split:

```bash
pip install -e ".[hf]"
python scripts/prepare_spider_ko.py \
  --out data/external/spider-ko-validation.jsonl \
  --manifest data/external/spider-ko-validation.manifest.json
```

Obtain the original Spider SQLite database directory separately and point `SPIDER_DB_ROOT` at its `database` directory.

## Smoke run

A smoke run proves wiring only. It is **never** portfolio evidence:

```bash
SMOKE=1 \
SPIDER_DB_ROOT=/path/to/spider/database \
bash scripts/run_spider_ko_colab.sh
```

The smoke path evaluates only 10 rows, so `portfolio_evidence_ready=false`.

## Full GPU run

A full run requires persistent storage. This is intentional: the previous Qwen adaptation experiment completed successfully but its raw JSON/ZIP was lost when the Colab runtime reset before download.

Example after mounting Google Drive in Colab:

```bash
ARCHIVE_DIR=/content/drive/MyDrive/aegis-spider-ko \
SPIDER_DB_ROOT=/content/spider/database \
bash scripts/run_spider_ko_colab.sh
```

Default model: `Qwen/Qwen2.5-Coder-1.5B-Instruct` in NF4 4-bit inference. `MODEL` and `ADAPTER` may be overridden, but different model/adapter runs must be reported separately.

## Evidence gate

A report is marked `portfolio_evidence_ready=true` only when all of these are true:

1. benchmark id is exactly `huggingface-KREW/spider-ko`
2. split is `validation`
3. question language is `ko`
4. all **1,034** validation examples were evaluated
5. all 1,034 row-level records are present
6. every row preserves `db_id`, Korean question, gold SQL, predicted SQL, and correctness

The report additionally keeps per-item latency and execution errors, dataset SHA-256, per-database SHA-256, source Git SHA, model/adapter identity, quantization, runtime versions, and GPU metadata.

## Interpretation rules

Do not claim the score as official Spider leaderboard accuracy. This runner uses the original per-database SQLite files but AEGIS's execution-result comparator rather than Spider's official evaluator. Its purpose is a reproducible **external generalisation check** inside this project.

If the result is poor, that is still useful evidence. It separates domain-specialised performance from cross-domain generalisation and gives a concrete basis for the next ablation: schema representation, model scale, adaptation data, or decoding strategy. A larger model should only be tested after the 1.5B external baseline is archived successfully.
