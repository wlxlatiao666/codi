#!/usr/bin/env bash
set -euo pipefail
export NO_COLOR=1 TERM=dumb HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1
cd /home/tiger/codi-ab-c71db6e-20260909/repository
exec /home/tiger/codi-ab-c71db6e-20260909/venv/bin/python -u scripts/cut_probe/run.py --output /home/tiger/codi-ab-c71db6e-20260909/fixed-codi-cut-probe-20261021 --model fixed
