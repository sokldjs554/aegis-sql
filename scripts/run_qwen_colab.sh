#!/usr/bin/env bash
set -euo pipefail

# Full LitE-SQL-inspired Qwen baseline run for a CUDA Colab/runtime.
# Usage from the repository root:
#   bash scripts/run_qwen_colab.sh
# Optional smoke mode:
#   SMOKE=1 bash scripts/run_qwen_colab.sh

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
SMOKE="${SMOKE:-0}"

if [[ "$SMOKE" == "1" ]]; then
  RUN_KIND="smoke"
  DEFAULT_OUT="data/generated/hf/qwen2.5-coder-1.5b-smoke"
  TRAIN_ARGS=(--limit-train 64 --limit-dev 32 --epochs 0.2)
  EVAL_ARGS=(--limit 10)
  EXPECTED_ITEMS=10
  echo "Running SMOKE mode: 64 train / 32 dev / 10 eval items"
else
  RUN_KIND="full"
  DEFAULT_OUT="data/generated/hf/qwen2.5-coder-1.5b-full"
  TRAIN_ARGS=()
  EVAL_ARGS=()
  EXPECTED_ITEMS=90
  echo "Running FULL mode: frozen 9,000-row Qwen snapshot + all 90 answerable KorFin items"
fi

OUT="${OUT:-$DEFAULT_OUT}"
BASE_REPORT="${BASE_REPORT:-reports/qwen2.5-coder-1.5b-${RUN_KIND}-base.json}"
ADAPTED_REPORT="${ADAPTED_REPORT:-reports/qwen2.5-coder-1.5b-${RUN_KIND}-adapted.json}"
SUMMARY_REPORT="${SUMMARY_REPORT:-reports/qwen2.5-coder-1.5b-${RUN_KIND}-summary.json}"
RESULT_BUNDLE="${RESULT_BUNDLE:-reports/qwen2.5-coder-1.5b-${RUN_KIND}-results.zip}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-data/research/qwen_flywheel_v1}"
REFERENCE_ROOT="${REFERENCE_ROOT:-data/generated/qwen-flywheel-v1}"
REFERENCE_DB="${REFERENCE_DB:-$REFERENCE_ROOT/aegis_demo.sqlite}"
REFERENCE_DATA="${REFERENCE_DATA:-$REFERENCE_ROOT/dataset}"

export PYTHONHASHSEED=0
export AEGIS_DATABASE__PATH="$REFERENCE_DB"
export AEGIS_FLYWHEEL__OUTPUT_DIR="$REFERENCE_DATA"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: NVIDIA GPU runtime is required. In Colab choose Runtime > Change runtime type > GPU." >&2
  exit 2
fi

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Select a GPU runtime before running the experiment.")
print("gpu:", torch.cuda.get_device_name(0))
PY

# Colab already ships torch. Install only the experiment stack so the runtime's
# CUDA-compatible torch build is not needlessly replaced.
python -m pip install -q --upgrade pip
python -m pip install -q -e ".[hf]"

# Materialize the committed synthetic snapshot instead of regenerating the
# flywheel in Colab. The old 12,540-row artefact was never committed and its
# profile cache did not include a database-content hash, so it cannot support
# an exact paired comparison. This snapshot makes every Qwen run use identical
# train/dev records; the historical AegisLM result remains explicitly unpaired.
mkdir -p "$REFERENCE_ROOT"
python scripts/build_demo_db.py --out "$REFERENCE_DB" --scale 1.0 --force
python scripts/qwen_dataset_snapshot.py materialize \
  --snapshot "$SNAPSHOT_DIR" \
  --out "$REFERENCE_DATA"

if [[ ! -f data/benchmark/korfin_bench.jsonl ]]; then
  python scripts/build_benchmark.py
fi

mkdir -p reports "$OUT"

# 0) Freeze the exact data/schema inputs before downloading the model.
python scripts/train_hf_text2sql.py \
  --model "$MODEL" \
  --qlora \
  --data-dir "$REFERENCE_DATA" \
  --seed 20260824 \
  --out "$OUT" \
  --prepare-only \
  "${TRAIN_ARGS[@]}"

# 1) Base model, before AEGIS-domain adaptation.
python scripts/eval_hf_text2sql.py \
  --model "$MODEL" \
  --load-4bit \
  --out "$BASE_REPORT" \
  "${EVAL_ARGS[@]}"

# 2) Same model adapted on the unchanged AEGIS flywheel split.
python scripts/train_hf_text2sql.py \
  --model "$MODEL" \
  --qlora \
  --data-dir "$REFERENCE_DATA" \
  --seed 20260824 \
  --out "$OUT" \
  "${TRAIN_ARGS[@]}"

# 3) Same evaluator, retrieval, guard, database and EX metric.
python scripts/eval_hf_text2sql.py \
  --model "$MODEL" \
  --adapter "$OUT/adapter" \
  --load-4bit \
  --out "$ADAPTED_REPORT" \
  "${EVAL_ARGS[@]}"

python scripts/summarize_qwen_experiment.py \
  --manifest "$OUT/experiment_manifest.json" \
  --base "$BASE_REPORT" \
  --adapted "$ADAPTED_REPORT" \
  --out "$SUMMARY_REPORT" \
  --run-kind "$RUN_KIND" \
  --expected-items "$EXPECTED_ITEMS"

python - "$RESULT_BUNDLE" "$SUMMARY_REPORT" "$BASE_REPORT" "$ADAPTED_REPORT" \
  "$OUT/experiment_manifest.json" <<'PY'
import sys
import zipfile
from pathlib import Path

bundle = Path(sys.argv[1])
artifacts = [Path(value) for value in sys.argv[2:]]
with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for artifact in artifacts:
        archive.write(artifact, arcname=artifact.name)
print(f"result bundle: {bundle}")
PY
