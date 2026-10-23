#!/usr/bin/env bash
set -euo pipefail
export NO_COLOR=1 TERM=dumb HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export CODI_ASSET_ROOT="${CODI_ASSET_ROOT:-/home/tiger/codi-ab-c71db6e-20260909}"
export ALIGNED_STEP_ROOT="${ALIGNED_STEP_ROOT:-$CODI_ASSET_ROOT/aligned-step-20261021}"
export ALIGNED_STEP_CHUNK="${ALIGNED_STEP_CHUNK:-64}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON_BIN="$CODI_ASSET_ROOT/venv/bin/python"
mkdir -p "$ALIGNED_STEP_ROOT"
"$PYTHON_BIN" scripts/reproduce/preflight.py --method continuation
CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" -m pytest -q tests/test_aligned_step.py tests/test_all_cuts.py tests/test_continuation.py
"$PYTHON_BIN" scripts/aligned_step/prepare.py
"$PYTHON_BIN" -u scripts/aligned_step/controller.py --smoke
exec "$PYTHON_BIN" -u scripts/aligned_step/controller.py
