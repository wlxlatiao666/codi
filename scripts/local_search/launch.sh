#!/usr/bin/env bash
# Run on the authorized GPU worker. All artifacts live outside the repository.
set -euo pipefail
export NO_COLOR=1 TERM=dumb HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CODI_ASSET_ROOT="${CODI_ASSET_ROOT:-/home/tiger/codi-ab-c71db6e-20260909}"
export CODI_SEARCH_ROOT="${CODI_SEARCH_ROOT:-$CODI_ASSET_ROOT/random-disturb-20261015}"
PYTHON_BIN="${CODI_PYTHON:-$CODI_ASSET_ROOT/venv/bin/python}"
cd "$REPO_DIR"
mkdir -p "$CODI_SEARCH_ROOT"
"$PYTHON_BIN" scripts/reproduce/preflight.py --method continuation
"$PYTHON_BIN" -m pytest -q tests/test_local_search_policy.py tests/test_continuation.py
if [[ ! -f "$CODI_SEARCH_ROOT/data-audit.json" ]]; then
  "$PYTHON_BIN" scripts/local_search/prepare.py
fi
if [[ ! -f "$CODI_SEARCH_ROOT/smoke-verification.json" ]]; then
  "$PYTHON_BIN" scripts/local_search/controller.py --smoke
fi
exec "$PYTHON_BIN" -u scripts/local_search/controller.py
