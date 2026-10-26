"""Train one (k=3,r=1/2) policy from scratch, with epoch checkpoints and OOD eval."""
import fcntl
import os
import sys
import traceback
from pathlib import Path
os.environ['CODI_SEARCH_ROOT'] = os.environ['SINGLE_CUT_ROOT']
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'local_search'))
import controller as lifecycle
from runtime import EXP, write_json
lifecycle.SCRIPTS = Path(__file__).resolve().parent
if __name__ == '__main__':
    smoke = '--smoke' in sys.argv
    with (EXP / 'train.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            write_json(EXP / 'status.json', dict(stage='smoke' if smoke else 'training'))
            last = lifecycle.run_phase('smoke' if smoke else 'full', [[3,3]], 1 if smoke else 40, 0, smoke=smoke)
            if not smoke:
                results = {split:lifecycle.evaluate(last['checkpoint'], EXP/'final'/split, split)
                           for split in ['validation','test','gsm-hard','multi-arith-full','svamp']}
                write_json(EXP / 'results.json', results)
            write_json(EXP / 'status.json',dict(stage='smoke_complete' if smoke else 'complete',checkpoint=last))
        except BaseException:
            write_json(EXP / 'status.json',dict(stage='failed',traceback=traceback.format_exc()))
            raise
