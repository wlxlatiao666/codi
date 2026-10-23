#!/usr/bin/env bash
set -euo pipefail
export NO_COLOR=1 TERM=dumb HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export CODI_ASSET_ROOT="${CODI_ASSET_ROOT:-/home/tiger/codi-ab-c71db6e-20260909}"
export WINDOW_SUFFIX_ROOT="${WINDOW_SUFFIX_ROOT:-$CODI_ASSET_ROOT/window-suffix-20261022}"
export WINDOW_SUFFIX_CHUNK="${WINDOW_SUFFIX_CHUNK:-64}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON_BIN="$CODI_ASSET_ROOT/venv/bin/python"
mkdir -p "$WINDOW_SUFFIX_ROOT"
"$PYTHON_BIN" scripts/reproduce/preflight.py --method continuation
CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" -m pytest -q tests/test_window_suffix.py tests/test_eval_routing.py tests/test_all_cuts.py tests/test_continuation.py
"$PYTHON_BIN" scripts/window_suffix/prepare.py
# All GPU3 evaluations share the existing all-cuts lock.
ln -sfn "$CODI_ASSET_ROOT/all-cuts-20261020/eval.lock" "$WINDOW_SUFFIX_ROOT/eval.lock"
"$PYTHON_BIN" -u scripts/window_suffix/controller.py --smoke
exec "$PYTHON_BIN" -u scripts/window_suffix/controller.py
