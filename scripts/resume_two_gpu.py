"""Resume the two interrupted runs on a two-GPU worker, preserving training config."""
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('CODI_ASSET_ROOT', '/home/tiger/codi-ab-c71db6e-20260909'))
METHOD = sys.argv[1]
assert METHOD in ('window', 'sim')
EXP = ROOT / ('window-suffix-20261022' if METHOD == 'window' else 'sim-retrain-20261023')
os.environ.update(CODI_SEARCH_ROOT=str(EXP), CODI_RUN_ROOT=str(EXP),
                  WINDOW_SUFFIX_ROOT=str(EXP), WINDOW_SUFFIX_CHUNK='64',
                  METHOD='sim' if METHOD == 'sim' else 'continuation',
                  CODI_EVAL_GPU='1', SIM_TRAIN_MEMORY_FRACTION='0.8',
                  HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                  TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1')

def status(stage, **kwargs):
    (EXP / 'resume-worker-4442481.json').write_text(json.dumps(dict(
        stage=stage, worker=4442481, train_gpu=0 if METHOD == 'window' else 1,
        eval_gpu=1, updated=time.time(), **kwargs), indent=2))

lockname = 'train.lock' if METHOD == 'window' else 'sim-train.lock'
with (EXP / lockname).open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        run = EXP / 'runs' / ('full' if METHOD == 'window' else 'sim')
        last = json.loads((run / 'last-checkpoint.json').read_text())
        assert last['epoch'] < 40
        cp = Path(last['checkpoint'])
        for filename in ['pytorch_model.bin', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json']:
            assert (cp / filename).stat().st_size > 0
        assert (EXP / 'eval.lock').resolve() == (ROOT / 'all-cuts-20261020/eval.lock').resolve()
        status('resuming', source_checkpoint=str(cp), source_epoch=last['epoch'])
        subprocess.run([sys.executable, str(REPO/'scripts/reproduce/preflight.py'),
                        '--method', 'continuation' if METHOD == 'window' else 'sim'], check=True)
        if METHOD == 'window':
            sys.path.insert(0, str(REPO/'scripts/local_search'))
            import controller as lifecycle
            lifecycle.SCRIPTS = REPO/'scripts/window_suffix'
            original_environment = lifecycle.environment
            def environment(gpu):
                # Keep original logical IDs in the immutable phase manifest.
                env = original_environment({2: 0, 3: 1}[gpu])
                env['SEARCH_EVAL_GPU'] = '1'
                return env
            lifecycle.environment = environment
            endpoint = lifecycle.run_phase('full',
                'continuous <=3 suffix starts around ceil(k*m/6); equal k weight 0.05', 40, 2)
            results = {split: lifecycle.evaluate(endpoint['checkpoint'], EXP/'final'/split, split)
                       for split in ['validation', 'test', 'gsm-hard', 'multi-arith-full', 'svamp']}
        else:
            sys.path.insert(0, str(REPO/'scripts/reproduce'))
            import controller as lifecycle
            lifecycle.SCRIPTS = REPO/'scripts/sim_retrain'
            lifecycle.status('training_40epoch')
            lifecycle.run('sim', 1)
            endpoint = json.loads((run/'last-checkpoint.json').read_text())
            results = {}
            for split in ['validation', 'test', 'gsm-hard', 'multi-arith-full', 'svamp']:
                dest = EXP/'final'/split
                dest.mkdir(parents=True, exist_ok=True)
                if not (dest/'metrics.json').exists():
                    script = 'evaluate.py' if split in ('validation', 'test') else 'evaluate_ood.py'
                    env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = '1'
                    with (EXP/'eval.lock').open('a') as elock, (dest/'eval.log').open('w') as log:
                        fcntl.flock(elock, fcntl.LOCK_EX)
                        subprocess.run([sys.executable, str(REPO/'scripts/reproduce'/script),
                            '--checkpoint', endpoint['checkpoint'], '--output', str(dest),
                            '--split', split, '--sim'], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                results[split] = json.loads((dest/'metrics.json').read_text())
            lifecycle.status('complete', checkpoint=endpoint['checkpoint'])
        (EXP/'results.json').write_text(json.dumps(results, indent=2))
        (EXP/'status.json').write_text(json.dumps(dict(stage='complete', checkpoint=endpoint), indent=2))
        status('complete', checkpoint=endpoint)
    except BaseException:
        status('failed', traceback=traceback.format_exc())
        raise
