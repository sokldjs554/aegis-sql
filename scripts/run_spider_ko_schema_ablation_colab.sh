#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
ARCHIVE_DIR="${ARCHIVE_DIR:-}"
RUN_DIR="${RUN_DIR:-}"
FORCE="${FORCE:-0}"
STYLES=(slm ddl compact mschema)
EXPECTED_ITEMS=1034
SPIDER_DB_DIR="${SPIDER_DB_DIR:-data/external/spider-db}"
SPIDER_DB_ROOT="${SPIDER_DB_ROOT:-}"
DATASET_FILE="${DATASET_FILE:-data/external/spider-ko-validation.jsonl}"
DATASET_MANIFEST="${DATASET_MANIFEST:-data/external/spider-ko-validation.manifest.json}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: an NVIDIA GPU runtime is required." >&2
  exit 2
fi
if [[ -z "$ARCHIVE_DIR" ]]; then
  echo "ERROR: full runs require ARCHIVE_DIR on persistent storage." >&2
  exit 2
fi
mkdir -p "$ARCHIVE_DIR"
if [[ ! -w "$ARCHIVE_DIR" ]]; then
  echo "ERROR: ARCHIVE_DIR is not writable: $ARCHIVE_DIR" >&2
  exit 2
fi

CURRENT_GIT_SHA="$(git rev-parse HEAD)"
if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$ARCHIVE_DIR/spider-ko-schema-ablation"
fi

if [[ "$FORCE" == "1" && -d "$RUN_DIR" ]]; then
  echo "FORCE=1: clearing prior ablation directory: $RUN_DIR"
  rm -rf "$RUN_DIR"
fi
mkdir -p "$RUN_DIR"

python -m pip install -q --upgrade pip
python -m pip install -q -e ".[hf]"

RUN_META="$RUN_DIR/run-meta.json"
python - "$RUN_META" "$CURRENT_GIT_SHA" "$MODEL" <<'PY'
import importlib.metadata
import json
import sys
from pathlib import Path

import torch

path = Path(sys.argv[1])
git_sha = sys.argv[2]
model = sys.argv[3]
packages = {}
for name in ("transformers", "peft", "accelerate", "bitsandbytes"):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
current = {
    "git_sha": git_sha,
    "model": model,
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "packages": packages,
}
if path.exists():
    previous = json.loads(path.read_text(encoding="utf-8"))
    if previous != current:
        raise SystemExit(
            "Existing ablation uses different git/runtime provenance. "
            "Use a new RUN_DIR or set FORCE=1 to rerun all four styles."
        )
else:
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(current, ensure_ascii=False, indent=2))
PY

if [[ -z "$SPIDER_DB_ROOT" ]]; then
  python scripts/prepare_spider_databases.py --out "$SPIDER_DB_DIR"
  SPIDER_DB_ROOT="$SPIDER_DB_DIR/spider_data/database"
fi
if [[ ! -d "$SPIDER_DB_ROOT" ]]; then
  echo "ERROR: Spider database directory does not exist: $SPIDER_DB_ROOT" >&2
  exit 2
fi
if [[ ! -f "$DATASET_FILE" || ! -f "$DATASET_MANIFEST" ]]; then
  python scripts/prepare_spider_ko.py \
    --out "$DATASET_FILE" \
    --manifest "$DATASET_MANIFEST"
fi

is_complete_report() {
  local report="$1"
  local style="$2"
  python - "$report" "$style" "$CURRENT_GIT_SHA" "$MODEL" "$EXPECTED_ITEMS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
style = sys.argv[2]
git_sha = sys.argv[3]
model = sys.argv[4]
expected = int(sys.argv[5])
try:
    report = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
evaluation = report.get("evaluation") or {}
rows = report.get("rows")
valid = (
    report.get("portfolio_evidence_ready") is True
    and report.get("items") == expected
    and isinstance(rows, list)
    and len(rows) == expected
    and report.get("model") == model
    and evaluation.get("schema_style") == style
    and evaluation.get("git_sha") == git_sha
    and evaluation.get("quantization") == "NF4 4-bit"
)
raise SystemExit(0 if valid else 1)
PY
}

for style in "${STYLES[@]}"; do
  REPORT="$RUN_DIR/spider-ko-$style.json"
  if [[ -f "$REPORT" ]] && is_complete_report "$REPORT" "$style"; then
    echo "[$style] complete report already archived; resume skip"
    continue
  fi
  rm -f "$REPORT"
  echo "[$style] running $EXPECTED_ITEMS Spider-KO validation rows"
  python scripts/eval_spider_ko_hf.py \
    --dataset "$DATASET_FILE" \
    --db-root "$SPIDER_DB_ROOT" \
    --model "$MODEL" \
    --schema-style "$style" \
    --load-4bit \
    --out "$REPORT"
  if ! is_complete_report "$REPORT" "$style"; then
    echo "ERROR: $style did not produce a complete portfolio_evidence_ready report." >&2
    exit 3
  fi
  echo "[$style] archived immediately: $REPORT"
done

SUMMARY="$RUN_DIR/spider-ko-schema-ablation-summary.json"
python scripts/summarize_spider_ko_schema_ablation.py \
  --slm "$RUN_DIR/spider-ko-slm.json" \
  --ddl "$RUN_DIR/spider-ko-ddl.json" \
  --compact "$RUN_DIR/spider-ko-compact.json" \
  --mschema "$RUN_DIR/spider-ko-mschema.json" \
  --expected-items "$EXPECTED_ITEMS" \
  --out "$SUMMARY"

cp "$DATASET_MANIFEST" "$RUN_DIR/spider-ko-dataset.manifest.json"
if [[ -f "$SPIDER_DB_DIR/spider-databases.manifest.json" ]]; then
  cp "$SPIDER_DB_DIR/spider-databases.manifest.json" "$RUN_DIR/spider-databases.manifest.json"
fi

echo "schema ablation complete: $SUMMARY"
