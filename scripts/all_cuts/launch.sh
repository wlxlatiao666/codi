#!/usr/bin/env bash
set -euo pipefail
export NO_COLOR=1 TERM=dumb HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export CODI_ASSET_ROOT="${CODI_ASSET_ROOT:-/home/tiger/codi-ab-c71db6e-20260909}"
export ALL_CUTS_ROOT="${ALL_CUTS_ROOT:-$CODI_ASSET_ROOT/all-cuts-20261020}"
export ALL_CUTS_CHUNK="${ALL_CUTS_CHUNK:-64}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON_BIN="$CODI_ASSET_ROOT/venv/bin/python"
mkdir -p "$ALL_CUTS_ROOT"
"$PYTHON_BIN" scripts/reproduce/preflight.py --method continuation
CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" -m pytest -q tests/test_all_cuts.py tests/test_continuation.py
"$PYTHON_BIN" scripts/all_cuts/prepare.py
"$PYTHON_BIN" -u scripts/all_cuts/controller.py --smoke
exec "$PYTHON_BIN" -u scripts/all_cuts/controller.py
