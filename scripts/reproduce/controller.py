"""One bounded official-source run; smoke uses separate output and no shared weights."""
import os,sys,json,time,subprocess,traceback,argparse
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
ROOT=Path(os.environ.get('CODI_ASSET_ROOT','/home/tiger/codi-ab-c71db6e-20260909'))
EXP=Path(os.environ.get('CODI_RUN_ROOT',str(ROOT/'reorganized-runs-20261014')))
SCRIPTS=REPO/'scripts/reproduce';PY=Path(sys.executable)
def status(stage,**kw):
 d=dict(stage=stage,updated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
 (EXP/(os.environ['METHOD']+'-controller-status.json')).write_text(json.dumps(d,indent=2));print(d,flush=True)
def run(method,gpu,smoke=False,stop=40):
 subprocess.run([str(PY),str(SCRIPTS/'preflight.py'),'--method',method],check=True)
 if smoke and not (EXP/'smoke-dataset.pt').exists():
  raise FileNotFoundError('Create a separate 256-row smoke cache before using --smoke')
 r=EXP/'runs'/(('smoke-' if smoke else '')+method);r.mkdir(parents=True,exist_ok=True)
 last=json.loads((r/'last-checkpoint.json').read_text()) if (r/'last-checkpoint.json').exists() else None
 if last and last['epoch']>=stop:return
 env=os.environ.copy();env.update(NO_COLOR='1',TERM='dumb',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES=str(gpu),METHOD=method,CODI_DATA_CACHE=str(EXP/'smoke-dataset.pt' if smoke else ROOT/('continuation-search-20260930/runs/random-cut/dataset.pt' if method=='continuation' else 'continuation-search-20260930/runs/base-codi/dataset.pt')),RUN_DIR=str(r),STOP_EPOCH=str(stop))
 for k in ['RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK','RESUME','SMOKE']:env.pop(k,None)
 if last:env['RESUME']=last['checkpoint']
 if smoke:env['SMOKE']='1'
 cmd=[str(PY),str(SCRIPTS/'train_entry.py'),'--output_dir',str(r),'--expt_name','run','--model_name_or_path',str(ROOT/'assets/models/gpt2'),'--data_name','icot',
 '--seed','11','--model_max_length','512','--per_device_train_batch_size','64','--gradient_accumulation_steps','2','--bf16','True','--num_train_epochs','40','--learning_rate','0.003','--max_grad_norm','2',
 '--use_lora','True','--lora_init','True','--lora_r','128','--lora_alpha','32','--save_strategy','no','--save_total_limit','2','--save_safetensors','False','--weight_decay','0.1','--warmup_ratio','0.03','--lr_scheduler_type','cosine',
 '--report_to','none','--logging_steps','20','--disable_tqdm','True','--num_latent','6','--use_prj','True','--prj_dim','768','--prj_dropout','0','--distill_loss_type','smooth_l1','--distill_loss_div_std','True','--distill_loss_factor','1','--ref_loss_factor','1','--remove_eos','True','--fix_attn_mask','False','--include_last_cot','False',
 '--use_decoder',str(method=='sim'),'--continuation_weight',str(.3 if method=='continuation' else 0.),'--explain_loss_factor','1','--print_ref_model_stats','False','--print_loss','True','--remove_unused_columns','False']
 if smoke:cmd+=['--exp_mode','True','--exp_data_num','256','--max_steps','4','--logging_steps','1']
 (r/f'phase-{stop:g}-launch.json').write_text(json.dumps(dict(command=cmd,resume=env.get('RESUME'),gpu=gpu,started=time.time()),indent=2))
 with (r/f'phase-{stop:g}.log').open('a') as log:
  p=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT);(r/'pid').write_text(str(p.pid));rc=p.wait()
 (r/f'phase-{stop:g}.exit').write_text(str(rc))
 if rc:raise RuntimeError(f'Official training failed ({rc}); inspect {r}/phase-{stop:g}.log')
 if not smoke:
  env['CUDA_VISIBLE_DEVICES']=os.environ.get('CODI_EVAL_GPU','3');dest=r/'test'
  import fcntl
  with (EXP/'eval.lock').open('w') as lock, (r/'test.log').open('w') as f:
   fcntl.flock(lock,fcntl.LOCK_EX)
   subprocess.run([str(PY),str(SCRIPTS/'evaluate.py'),'--checkpoint',json.loads((r/'last-checkpoint.json').read_text())['checkpoint'],'--output',str(dest),'--split','test']+(['--sim'] if method=='sim' else []),env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
if __name__=='__main__':
 EXP.mkdir(parents=True,exist_ok=True)
 import concurrent.futures
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--method',choices=['codi','sim','continuation'],required=True);ap.add_argument('--gpu',type=int,required=True);a=ap.parse_args()
 def one(pair):
  method,gpu=pair
  try:
   if a.smoke: run(method,gpu,True,1);run(method,gpu,True,2)
   else: run(method,gpu)
   (EXP/f'{method}-status.json').write_text(json.dumps(dict(stage='smoke_complete' if a.smoke else 'complete',updated=time.time())))
  except Exception as error:
   (EXP/f'{method}-status.json').write_text(json.dumps(dict(stage='failed',error=str(error),updated=time.time())))
   raise
 try:
  status('smoke' if a.smoke else 'training_40epoch')
  import fcntl
  with (EXP/(a.method+'-train.lock')).open('w') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
   one((a.method,a.gpu))
  status('smoke_complete' if a.smoke else 'complete')
 except Exception as e:status('failed',error=str(e),traceback=traceback.format_exc());raise
