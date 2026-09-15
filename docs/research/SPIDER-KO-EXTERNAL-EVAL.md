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

## Archived 1.5B baseline — 2026-09-15

The first complete external baseline was archived from a Tesla T4 run with no adapter:

- model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- schema style: `slm`
- quantization: NF4 4-bit
- evaluated rows: **1,034 / 1,034**
- AEGIS execution match: **369 / 1,034 = 35.7%**
- execution failures recorded by the evaluator: **337**
- latency: **p50 2,375.84 ms / p95 4,263.76 ms** (`model.generate` only, CUDA synchronized)
- report Git SHA: `8be61dbc0b8146df094eebcefe9e5ef7ce66e391`
- dataset SHA-256: `db802bab717deb2e16f2fa6cbb493712ac294515e68a69f1a243c5238cf99b52`
- `portfolio_evidence_ready=true`

The row-level audit showed that execution failure is dominated by schema-reference errors: **312 `no such column`** and **7 `no such table`** prediction errors. Two WTA rows fail for both prediction and gold because the SQLite text cannot be decoded as UTF-8, so the headline execution-failure count must not be described as 337 pure model failures.

This baseline is retained as historical evidence. It is **not** silently reused as the `slm` arm of later ablations because the evaluator Git SHA and runtime provenance must be identical across all arms.

## Schema representation ablation

The next controlled experiment keeps the model, decoding, quantization, dataset, database files, evaluator commit, package versions, and GPU identity fixed while changing only the schema representation:

- `slm`
- `ddl`
- `compact`
- `mschema`

Run it from the Colab notebook:

`notebooks/spider_ko_schema_ablation.ipynb`

or after mounting Google Drive:

```bash
ARCHIVE_DIR=/content/drive/MyDrive/AEGIS-SQL \
bash scripts/run_spider_ko_schema_ablation_colab.sh
```

The runner writes each completed 1,034-row report directly to persistent storage before starting the next style. If Colab disconnects, a rerun skips complete styles. `run-meta.json` pins the experiment to one Git SHA and runtime stack; a changed runtime is rejected instead of mixing incomparable evidence. Set `FORCE=1` only when intentionally discarding the entire prior four-arm run.

After all four reports are complete, `scripts/summarize_spider_ko_schema_ablation.py` verifies provenance equality and writes `spider-ko-schema-ablation-summary.json`. The comparison reports:

- execution accuracy and delta vs `slm`
- total execution failures
- prediction/gold execution failures separately
- `no such column`
- `no such table`
- combined schema-reference failures
- p50 / p95 generation latency

The winner is selected by execution accuracy first, then fewer schema-reference failures, fewer execution failures, and lower p95 latency as tie-breakers.

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

Do not merge KorFin-Bench and Spider-KO into one headline metric. The former measures the project-specific finance/insurance system; the latter measures external cross-domain generalisation.

Model scale is not the next variable while schema-reference failures dominate the 1.5B baseline. First measure whether schema representation reduces those failures. Then test a bounded execution-guided repair on the best representation. Move to 3B only if compositional reasoning failures remain after those controls; preserve the 1.5B baseline throughout.
