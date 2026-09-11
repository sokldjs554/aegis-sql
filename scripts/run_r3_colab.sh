#!/usr/bin/env bash
set -euo pipefail

# R³-SQL-inspired paired selective-resampling run.
#
# Hosted full run (put the key in a Colab secret/environment variable first):
#   MAX_COST_USD=5 bash scripts/run_r3_colab.sh
#
# Offline wiring smoke:
#   PROVIDER=mock LIMIT=3 bash scripts/run_r3_colab.sh
#
# To survive a Colab VM reset, mount Drive and point PREFIX into it:
#   PREFIX=/content/drive/MyDrive/aegis-sql-evidence/r3-selective-resampling \
#   MAX_COST_USD=5 bash scripts/run_r3_colab.sh

PROVIDER="${PROVIDER:-anthropic}"
ENSEMBLE_SAMPLES="${ENSEMBLE_SAMPLES:-5}"
MAX_COST_USD="${MAX_COST_USD:-}"
LIMIT="${LIMIT:-}"
RESUME="${RESUME:-0}"
PREFIX="${PREFIX:-reports/r3-selective-resampling}"
COLAB_DOWNLOAD="${COLAB_DOWNLOAD:-1}"

case "$PROVIDER" in
  anthropic)
    MODEL="${MODEL:-claude-sonnet-5}"
    KEY_NAME="ANTHROPIC_API_KEY"
    ;;
  openai)
    MODEL="${MODEL:-gpt-4o}"
    KEY_NAME="OPENAI_API_KEY"
    ;;
  mock)
    MODEL="${MODEL:-mock}"
    KEY_NAME=""
    ;;
  *)
    echo "ERROR: PROVIDER must be anthropic, openai, or mock" >&2
    exit 2
    ;;
esac

if [[ "$PROVIDER" != "mock" ]]; then
  if [[ -z "$MAX_COST_USD" ]]; then
    echo "ERROR: set MAX_COST_USD to the observed-cost stopping threshold (recommended first cap: 5)." >&2
    exit 2
  fi
  if [[ -z "${!KEY_NAME:-}" ]]; then
    echo "ERROR: $KEY_NAME is unset. Load it from Colab Secrets; never paste it into source or reports." >&2
    exit 2
  fi
fi

export PYTHONHASHSEED=0

python -m pip install -q --upgrade pip
python -m pip install -q -e ".[llm]"

# The SQLite database is generated, not committed. Rebuilding at the frozen
# scale and hash seed gives the manifest an auditable content hash.
python scripts/build_demo_db.py --out data/demo/aegis_demo.sqlite --scale 1.0 --force
if [[ ! -f data/benchmark/korfin_bench.jsonl ]]; then
  python scripts/build_benchmark.py
fi

ARGS=(
  --provider "$PROVIDER"
  --model "$MODEL"
  --ensemble-samples "$ENSEMBLE_SAMPLES"
  --prefix "$PREFIX"
)
if [[ -n "$MAX_COST_USD" ]]; then
  ARGS+=(--max-cost-usd "$MAX_COST_USD")
fi
if [[ -n "$LIMIT" ]]; then
  ARGS+=(--limit "$LIMIT")
fi
if [[ "$RESUME" == "1" ]]; then
  ARGS+=(--resume)
fi

set +e
python scripts/run_selective_resampling.py "${ARGS[@]}"
STATUS=$?
set -e

RESULT_BUNDLE="${PREFIX}-results.zip"
if [[ "$COLAB_DOWNLOAD" == "1" && -f "$RESULT_BUNDLE" ]]; then
  python - "$RESULT_BUNDLE" <<'PY'
import sys

try:
    from google.colab import files
except ImportError:
    print(f"result bundle: {sys.argv[1]}")
else:
    print(f"starting browser download: {sys.argv[1]}")
    files.download(sys.argv[1])
PY
fi

exit "$STATUS"
