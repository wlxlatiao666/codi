#!/usr/bin/env bash
set -euo pipefail
export NO_COLOR=1 TERM=dumb HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 METHOD=sim CODI_EVAL_GPU=3
export CODI_ASSET_ROOT="${CODI_ASSET_ROOT:-/home/tiger/codi-ab-c71db6e-20260909}"
export CODI_RUN_ROOT="${CODI_RUN_ROOT:-$CODI_ASSET_ROOT/sim-retrain-20261023}"
export SIM_TRAIN_MEMORY_FRACTION=0.8
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="$CODI_ASSET_ROOT/venv/bin/python"
mkdir -p "$CODI_RUN_ROOT"
"$PY" scripts/reproduce/preflight.py --method sim
"$PY" - <<'PY'
import os,json
from pathlib import Path
import torch
r=Path(os.environ['CODI_ASSET_ROOT']);e=Path(os.environ['CODI_RUN_ROOT']);lock=e/'eval.lock';target=r/'all-cuts-20261020/eval.lock'
assert target.exists()
if lock.exists() or lock.is_symlink():assert lock.resolve()==target.resolve()
else:lock.symlink_to(target)
p=e/'smoke-dataset.pt'
if not p.exists():
 data=torch.load(r/'continuation-search-20260930/runs/base-codi/dataset.pt',map_location='cpu',weights_only=False)
 sample={k:v[:256] for k,v in data.items()};torch.save(sample,p)
(e/'protocol.json').write_text(json.dumps(dict(method='CODI + SIM-CoT',gpu=3,eval_gpu=3,epochs=40,seed=11,batch=64,accumulation=2,lr=.003,explain_loss_factor=1,continuation_weight=0,use_decoder=True,num_latent=6,save_each_epoch=True,keep_latest=2,eval_every_epochs=4,allocator_fraction=.8,shared_eval_lock=str(target),source='unchanged scripts/reproduce/train_entry.py',recovery='new training; original weights not recovered'),indent=2))
PY
exec "$PY" -u scripts/sim_retrain/controller.py
