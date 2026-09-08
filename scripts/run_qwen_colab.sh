#!/usr/bin/env bash
set -euo pipefail

# Full LitE-SQL-inspired Qwen baseline run for a CUDA Colab/runtime.
# Usage from the repository root:
#   bash scripts/run_qwen_colab.sh
# Optional smoke mode:
#   SMOKE=1 bash scripts/run_qwen_colab.sh

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
OUT="${OUT:-data/generated/hf/qwen2.5-coder-1.5b}"
SMOKE="${SMOKE:-0}"

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

# The comparison must use the real AEGIS demo schema and the existing skeleton-
# clustered flywheel split. Build only when the artifacts are absent.
if [[ ! -f data/demo/aegis_demo.sqlite ]]; then
  python scripts/build_demo_db.py --scale 1.0
fi
if [[ ! -f data/benchmark/korfin_bench.jsonl ]]; then
  python scripts/build_benchmark.py
fi
if [[ ! -f data/generated/flywheel/train.jsonl || ! -f data/generated/flywheel/dev.jsonl ]]; then
  python -m aegis_sql.cli flywheel --n-programs 4000
fi

if [[ "$SMOKE" == "1" ]]; then
  TRAIN_ARGS=(--limit-train 64 --limit-dev 32 --epochs 0.2)
  EVAL_ARGS=(--limit 10)
  echo "Running SMOKE mode: 64 train / 32 dev / 10 eval items"
else
  TRAIN_ARGS=()
  EVAL_ARGS=()
  echo "Running FULL mode: existing full flywheel split + all answerable KorFin items"
fi

mkdir -p reports "$OUT"

# 0) Freeze the exact data/schema inputs before downloading the model.
python scripts/train_hf_text2sql.py \
  --model "$MODEL" \
  --qlora \
  --out "$OUT" \
  --prepare-only \
  "${TRAIN_ARGS[@]}"

# 1) Base model, before AEGIS-domain adaptation.
python scripts/eval_hf_text2sql.py \
  --model "$MODEL" \
  --load-4bit \
  --out reports/qwen2.5-coder-1.5b-base.json \
  "${EVAL_ARGS[@]}"

# 2) Same model adapted on the unchanged AEGIS flywheel split.
python scripts/train_hf_text2sql.py \
  --model "$MODEL" \
  --qlora \
  --out "$OUT" \
  "${TRAIN_ARGS[@]}"

# 3) Same evaluator, retrieval, guard, database and EX metric.
python scripts/eval_hf_text2sql.py \
  --model "$MODEL" \
  --adapter "$OUT/adapter" \
  --load-4bit \
  --out reports/qwen2.5-coder-1.5b-adapted.json \
  "${EVAL_ARGS[@]}"

python - <<'PY'
import json
from pathlib import Path

paths = {
    "base": Path("reports/qwen2.5-coder-1.5b-base.json"),
    "adapted": Path("reports/qwen2.5-coder-1.5b-adapted.json"),
}
rows = {}
for name, path in paths.items():
    d = json.loads(path.read_text(encoding="utf-8"))
    rows[name] = d
    print("\n", name.upper())
    print("items:", d["items"])
    print("EX:", f'{d["execution_accuracy"]:.1%}')
    print("difficulty:", {k: f'{v["ex"]:.1%}' for k, v in d["by_difficulty"].items()})
    print("latency_ms:", d["latency_ms"])
    if "max_cuda_memory_bytes" in d:
        print("peak_cuda_GiB:", round(d["max_cuda_memory_bytes"] / (1024**3), 2))

base = rows["base"]["execution_accuracy"]
adapted = rows["adapted"]["execution_accuracy"]
print("\nDELTA")
print("adapted - base:", f'{(adapted-base)*100:+.1f} percentage points')
print("\nRaw reports:")
for p in paths.values():
    print(" -", p)
print("Manifest:", "data/generated/hf/qwen2.5-coder-1.5b/experiment_manifest.json")
PY
