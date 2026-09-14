#!/usr/bin/env bash
set -euo pipefail

# External Korean Text-to-SQL generalisation run.
#
# Required:
#   SPIDER_DB_ROOT=/path/to/spider/database
# The root must contain <db_id>/<db_id>.sqlite from the official Spider data.
#
# Full runs also require a persistent archive directory.  In Colab mount Drive
# first and point ARCHIVE_DIR there so a runtime reset cannot erase row-level
# evidence as happened in the earlier Qwen adaptation experiment.
#
# Smoke:
#   SMOKE=1 SPIDER_DB_ROOT=/content/spider/database bash scripts/run_spider_ko_colab.sh
# Full:
#   ARCHIVE_DIR=/content/drive/MyDrive/aegis-spider-ko \
#   SPIDER_DB_ROOT=/content/spider/database bash scripts/run_spider_ko_colab.sh

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
ADAPTER="${ADAPTER:-}"
SMOKE="${SMOKE:-0}"
SPIDER_DB_ROOT="${SPIDER_DB_ROOT:-}"
DATASET_FILE="${DATASET_FILE:-data/external/spider-ko-validation.jsonl}"
DATASET_MANIFEST="${DATASET_MANIFEST:-data/external/spider-ko-validation.manifest.json}"
ARCHIVE_DIR="${ARCHIVE_DIR:-}"

if [[ -z "$SPIDER_DB_ROOT" ]]; then
  echo "ERROR: set SPIDER_DB_ROOT to the official Spider database directory." >&2
  exit 2
fi
if [[ ! -d "$SPIDER_DB_ROOT" ]]; then
  echo "ERROR: SPIDER_DB_ROOT does not exist: $SPIDER_DB_ROOT" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: an NVIDIA GPU runtime is required." >&2
  exit 2
fi

if [[ "$SMOKE" == "1" ]]; then
  RUN_KIND="smoke"
  LIMIT_ARGS=(--limit 10)
else
  RUN_KIND="full"
  LIMIT_ARGS=()
  if [[ -z "$ARCHIVE_DIR" ]]; then
    echo "ERROR: full runs require ARCHIVE_DIR on persistent storage." >&2
    exit 2
  fi
  mkdir -p "$ARCHIVE_DIR"
  if [[ ! -w "$ARCHIVE_DIR" ]]; then
    echo "ERROR: ARCHIVE_DIR is not writable: $ARCHIVE_DIR" >&2
    exit 2
  fi
fi

python -m pip install -q --upgrade pip
python -m pip install -q -e ".[hf]"

if [[ ! -f "$DATASET_FILE" || ! -f "$DATASET_MANIFEST" ]]; then
  python scripts/prepare_spider_ko.py \
    --out "$DATASET_FILE" \
    --manifest "$DATASET_MANIFEST"
fi

REPORT="reports/spider-ko-${RUN_KIND}.json"
BUNDLE="reports/spider-ko-${RUN_KIND}-results.zip"
mkdir -p reports

EVAL_ARGS=(
  --dataset "$DATASET_FILE"
  --db-root "$SPIDER_DB_ROOT"
  --model "$MODEL"
  --load-4bit
  --out "$REPORT"
)
if [[ -n "$ADAPTER" ]]; then
  EVAL_ARGS+=(--adapter "$ADAPTER")
fi
EVAL_ARGS+=("${LIMIT_ARGS[@]}")
python scripts/eval_spider_ko_hf.py "${EVAL_ARGS[@]}"

python - "$BUNDLE" "$REPORT" "$DATASET_MANIFEST" <<'PY'
import sys
import zipfile
from pathlib import Path

bundle = Path(sys.argv[1])
artifacts = [Path(value) for value in sys.argv[2:]]
with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for artifact in artifacts:
        if artifact.exists():
            archive.write(artifact, arcname=artifact.name)
print(f"result bundle: {bundle}")
PY

if [[ "$RUN_KIND" == "full" ]]; then
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  cp "$REPORT" "$ARCHIVE_DIR/spider-ko-full-$stamp.json"
  cp "$BUNDLE" "$ARCHIVE_DIR/spider-ko-full-$stamp.zip"
  cp "$DATASET_MANIFEST" "$ARCHIVE_DIR/spider-ko-dataset-$stamp.manifest.json"
  echo "persistent archive: $ARCHIVE_DIR"
fi
