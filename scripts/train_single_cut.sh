#!/usr/bin/env bash
set -euo pipefail
export NO_COLOR=1 TERM=dumb HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export CODI_ASSET_ROOT="${CODI_ASSET_ROOT:-/home/tiger/codi-ab-c71db6e-20260909}"
export SINGLE_CUT_ROOT="${SINGLE_CUT_ROOT:-$CODI_ASSET_ROOT/single-cut-k3-r50-20261023}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="$CODI_ASSET_ROOT/venv/bin/python"
mkdir -p "$SINGLE_CUT_ROOT"
"$PY" scripts/reproduce/preflight.py --method continuation
"$PY" - <<'PY'
import hashlib,json,os
from pathlib import Path
r=Path(os.environ['CODI_ASSET_ROOT']);e=Path(os.environ['SINGLE_CUT_ROOT'])
a=json.loads((r/'random-disturb-20261015/data-audit.json').read_text())
h=hashlib.sha256()
with (r/'continuation-dual-random-20261010/dataset.pt').open('rb') as f:
    for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
assert h.hexdigest()==a['bank_sha256']
for name,target in [('smoke-bank.pt',r/'random-disturb-20261015/smoke-bank.pt'),('eval.lock',r/'all-cuts-20261020/eval.lock')]:
    p=e/name
    assert target.exists(),target
    if p.exists() or p.is_symlink():assert p.resolve()==target.resolve(),p
    else:p.symlink_to(target)
(e/'protocol.json').write_text(json.dumps(dict(pairs=[[3,.5]],pairs_sixths=[[3,3]],continuation_weight=.3,epochs=40,seed=11,gpu=0,evaluation_gpu=3,bank_sha256=h.hexdigest(),main_data_sha256=a['main_data_sha256'],targets='EOT + steps j+1..m + answer + EOS; j=clamp(ceil(m/2),1,m-1)',total_updates=119840,main_samples=383368,eligible=a['eligible'],evaluation_every_epochs=4,save_every_epoch=True,resume='latest complete epoch including optimizer/scheduler/RNG'),indent=2))
PY
"$PY" -u scripts/single_cut/controller.py --smoke
exec "$PY" -u scripts/single_cut/controller.py
