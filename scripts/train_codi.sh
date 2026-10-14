#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CODI_ASSET_ROOT="${CODI_ASSET_ROOT:-/home/tiger/codi-ab-c71db6e-20260909}"
export CODI_RUN_ROOT="${CODI_RUN_ROOT:-$CODI_ASSET_ROOT/reorganized-runs-20261014}"
export CODI_EVAL_GPU="${CODI_EVAL_GPU:-3}"
export METHOD=codi
PYTHON_BIN="${CODI_PYTHON:-$CODI_ASSET_ROOT/venv/bin/python}"
cd "$REPO_DIR"
exec "$PYTHON_BIN" scripts/reproduce/controller.py --method codi --gpu "${GPU:-0}" "$@"
