"""Train local-window suffix supervision and evaluate the final checkpoint."""
import os
import sys
import json
import traceback
from pathlib import Path
os.environ['CODI_SEARCH_ROOT'] = os.environ['WINDOW_SUFFIX_ROOT']
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'local_search'))
import controller as lifecycle
from runtime import EXP, write_json
lifecycle.SCRIPTS = Path(__file__).resolve().parent
original_environment = lifecycle.environment
def environment(gpu):
    env = original_environment(gpu)
    env['SEARCH_EVAL_GPU'] = '3'
    return env
lifecycle.environment = environment
# run_phase expects train_phase.py; the small entry point delegates to train_run.
if __name__ == '__main__':
    smoke = '--smoke' in sys.argv
    try:
        write_json(EXP / 'status.json', dict(stage='smoke' if smoke else 'training'))
        result = lifecycle.run_phase('smoke' if smoke else 'full', 'continuous <=3 suffix starts around ceil(k*m/6); equal k weight 0.05',
                                     1 if smoke else 40, 2, smoke=smoke)
        if not smoke:
            results = {split: lifecycle.evaluate(result['checkpoint'], EXP / 'final' / split, split)
                       for split in ['validation', 'test', 'gsm-hard', 'multi-arith-full', 'svamp']}
            write_json(EXP / 'results.json', results)
        write_json(EXP / 'status.json', dict(stage='smoke_complete' if smoke else 'complete', checkpoint=result))
    except BaseException:
        write_json(EXP / 'status.json', dict(stage='failed', traceback=traceback.format_exc()))
        raise
