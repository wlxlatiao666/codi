#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CODI_ASSET_ROOT="${CODI_ASSET_ROOT:-/home/tiger/codi-ab-c71db6e-20260909}"
PYTHON_BIN="${CODI_PYTHON:-$CODI_ASSET_ROOT/venv/bin/python}"
METHOD="${1:?codi, sim, or continuation}"
CHECKPOINT="${2:?checkpoint directory}"
DEST="${3:?new evaluation output directory}"
case "$METHOD" in codi|sim|continuation) ;; *) exit 2;; esac
EXTRA=(); if [[ "$METHOD" == sim ]]; then EXTRA=(--sim); fi
export CUDA_VISIBLE_DEVICES="${GPU:-0}" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
"$PYTHON_BIN" "$REPO_DIR/scripts/reproduce/evaluate.py" --checkpoint "$CHECKPOINT" --output "$DEST/gsm8k-aug" --split test "${EXTRA[@]}"
for DATASET in gsm-hard multi-arith-full svamp; do
 "$PYTHON_BIN" "$REPO_DIR/scripts/reproduce/evaluate_ood.py" --checkpoint "$CHECKPOINT" --output "$DEST/$DATASET" --split "$DATASET" "${EXTRA[@]}"
done
