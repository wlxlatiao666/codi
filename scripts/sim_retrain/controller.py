"""Reproduce CODI+SIM-CoT on GPU3 with shared serialized evaluation."""
import fcntl,json,os,sys,subprocess,traceback
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts/reproduce'))
import controller as original
original.SCRIPTS=Path(__file__).resolve().parent
EXP=original.EXP
if __name__=='__main__':
 with (EXP/'sim-train.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   original.status('smoke')
   original.run('sim',3,smoke=True,stop=1)
   original.status('training_40epoch')
   original.run('sim',3)
   checkpoint=json.loads((EXP/'runs/sim/last-checkpoint.json').read_text())['checkpoint']
   results={}
   for split in ['validation','test','gsm-hard','multi-arith-full','svamp']:
    dest=EXP/'final'/split;dest.mkdir(parents=True,exist_ok=True)
    if not (dest/'metrics.json').exists():
     script='evaluate.py' if split in ['validation','test'] else 'evaluate_ood.py'
     env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='3'
     with (EXP/'eval.lock').open('a') as elock,(dest/'eval.log').open('w') as log:
      fcntl.flock(elock,fcntl.LOCK_EX)
      subprocess.run([sys.executable,str(REPO/'scripts/reproduce'/script),'--checkpoint',checkpoint,'--output',str(dest),'--split',split,'--sim'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    results[split]=json.loads((dest/'metrics.json').read_text())
   (EXP/'results.json').write_text(json.dumps(results,indent=2))
   original.status('complete',checkpoint=checkpoint)
  except BaseException:
   original.status('failed',traceback=traceback.format_exc());raise
