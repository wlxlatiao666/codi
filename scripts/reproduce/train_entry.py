"""Run the untouched cloned CODI/train.py with data I/O and lifecycle hooks only."""
import os,sys,json,time,hashlib,subprocess,math,inspect
from pathlib import Path
import torch
from datasets import load_from_disk
from transformers import set_seed,TrainerCallback
REPO=Path(__file__).resolve().parents[2]
ROOT=Path(os.environ.get('CODI_ASSET_ROOT','/home/tiger/codi-ab-c71db6e-20260909'))
EXP=Path(os.environ.get('CODI_RUN_ROOT',str(ROOT/'reorganized-runs-20261014')))
sys.path.insert(0,str(REPO))
import train
RUN=Path(os.environ['RUN_DIR']);SMOKE=os.environ.get('SMOKE')=='1'
STOP=float(os.environ.get('STOP_EPOCH','40'))
COMMIT='d1d56afbe705cfcbf5911b588da6c3083825598d'
set_seed(11)

# Only redirect the author's hardcoded, unavailable JSON path. Keep official
# row filtering, tokenization, collator, model constructor/forward and loss.
def local_data(path):
 assert path=='/mnt/shared-storage-user/weixilin/MLLM/coconut/data/gsm_train_clean.json',path
 return load_from_disk(str(ROOT/'continuation-search-20260930/data/train'))['train']
train.read_json=local_data

class Audit(TrainerCallback):
 def __init__(self,trainer):self.trainer=trainer;self.times=[]
 def on_train_begin(self,args,state,control,model=None,optimizer=None,**kw):
  assert args.seed==11 and args.num_train_epochs==40
  assert args.per_device_train_batch_size==64 and args.gradient_accumulation_steps==2
  assert args.learning_rate==.003 and args.num_latent==6 and not args.fix_attn_mask and not args.include_last_cot
  assert model.model_args.use_decoder == (os.environ['METHOD']=='sim')
  assert args.explain_loss_factor==1. and args.continuation_weight == (.3 if os.environ['METHOD']=='continuation' else 0.)
  assert Path(inspect.getfile(model.__class__)).resolve()==(REPO/'src/model.py').resolve()
  ds=self.trainer.train_dataset
  dh=hashlib.sha256()
  for k in ['encoder_input_ids','decoder_input_ids','ref_input_ids']:
   dh.update(k.encode())
   for t in ds.data_dict[k]:dh.update(t.numpy().tobytes())
  meta=dict(method=os.environ['METHOD'],official_base_commit=COMMIT,historical_core_byte_identical=True,model_file_sha256=hashlib.sha256((REPO/'src/model.py').read_bytes()).hexdigest(),train_file_sha256=hashlib.sha256((REPO/'train.py').read_bytes()).hexdigest(),main_data_sha256=dh.hexdigest(),train_samples=len(ds),max_steps=state.max_steps,seed=args.seed,batch=64,accumulation=2,learning_rate=args.learning_rate,fix_attn_mask=args.fix_attn_mask,explain_loss_factor=args.explain_loss_factor,scheduler_horizon_epochs=40,global_step=state.global_step,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
  if not SMOKE:
   ref=json.loads((REPO/'configs/reproduction.json').read_text())['training_reference']
   for k in ['main_data_sha256','train_samples','max_steps','seed','batch','accumulation','scheduler_horizon_epochs']:assert meta[k]==ref[k],(k,meta[k],ref[k])
  if not os.environ.get('RESUME'):
   assert state.global_step==0 and not optimizer.state
   h=hashlib.sha256();shared=hashlib.sha256()
   for n,t in model.state_dict().items():
    raw=t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    h.update(n.encode());h.update(raw)
    if not n.startswith('decoder.'):shared.update(n.encode());shared.update(raw)
   meta['initial_state_sha256']=h.hexdigest();meta['shared_initial_state_sha256']=shared.hexdigest()
   # Record rather than overwrite the official constructor's initialization.
   meta['initialization_policy']='seed11 before official constructor; official order unchanged'
   if not SMOKE:
    assert meta['shared_initial_state_sha256']==ref['initial_state_sha256'], 'Shared initialization mismatch'
    meta['shared_initialization_matches_continuation']=True
   (RUN/'fresh-start.json').write_text(json.dumps(meta,indent=2))
  else:
   assert state.global_step>0 and optimizer.state
   (RUN/'resume-audit.json').write_text(json.dumps(meta,indent=2))
  self.phase_start=time.time()
  print('OFFICIAL_AUDIT '+json.dumps(meta),flush=True)
 def on_log(self,args,state,control,logs=None,**kw):
  if not hasattr(self,'recent_metrics'): self.recent_metrics={}
  self.recent_metrics.update(logs or {})
  for k,v in (logs or {}).items():
   if k in {'loss','ce_loss','distill_loss','ref_ce_loss','explain_loss','continuation_loss','grad_norm'} and v is not None and not math.isfinite(float(v)):
    raise FloatingPointError(f'Nonfinite {k} at step {state.global_step}: {v}')
  (RUN/'progress.json').write_text(json.dumps(dict(global_step=state.global_step,max_steps=state.max_steps,epoch=state.epoch,updated=time.time(),metrics=self.recent_metrics),indent=2))
 def on_step_begin(self,args,state,control,**kw):
  if len(self.times)<30:torch.cuda.synchronize();self.start=time.time()
 def on_step_end(self,args,state,control,**kw):
  if len(self.times)<30:
   torch.cuda.synchronize();self.times.append(time.time()-self.start)
   if len(self.times)>=5:
    sec=sum(self.times[3:])/len(self.times[3:])
    (RUN/'throughput.json').write_text(json.dumps(dict(seconds_per_step=sec,estimated_40epoch_hours=sec*state.max_steps/3600,peak_cuda_gb=torch.cuda.max_memory_allocated()/2**30,global_step=state.global_step),indent=2))
 def on_epoch_end(self,args,state,control,**kw):
  control.should_save=True
  if state.epoch>=STOP-1e-6:control.should_training_stop=True
  return control
 def on_save(self,args,state,control,**kw):
  cp=Path(args.output_dir)/f'checkpoint-{state.global_step}';epoch=round(state.epoch,5)
  checkpoint=dict(epoch=epoch,checkpoint=str(cp),global_step=state.global_step)
  (RUN/'last-checkpoint.json').write_text(json.dumps(checkpoint,indent=2))
  if not SMOKE and round(epoch)%4: return
  dest=RUN/'validation'/f'epoch-{epoch:g}'
  cmd=[sys.executable,str(REPO/'scripts/reproduce/evaluate.py'),'--checkpoint',str(cp),'--output',str(dest)]
  if SMOKE:cmd+=['--limit','16']
  env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=os.environ.get('CODI_EVAL_GPU','3')
  if os.environ['METHOD']=='sim': cmd+=['--sim']
  import fcntl
  with (EXP/'eval.lock').open('w') as lock, (RUN/f'eval-{epoch:g}.log').open('w') as f:
   fcntl.flock(lock,fcntl.LOCK_EX)
   subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  checkpoint.update(metrics=json.loads((dest/'metrics.json').read_text()),phase_seconds=time.time()-self.phase_start)
  (RUN/'latest.json').write_text(json.dumps(checkpoint,indent=2))
  with (RUN/'history.jsonl').open('a') as f:f.write(json.dumps(checkpoint)+'\n')
  print('PHASE_EVAL '+json.dumps(checkpoint),flush=True)

original_train=train.CustomTrainer.train
def with_lifecycle(self,*a,**kw):
 self.add_callback(Audit(self))
 if os.environ.get('RESUME'):kw['resume_from_checkpoint']=os.environ['RESUME']
 return original_train(self,*a,**kw)
train.CustomTrainer.train=with_lifecycle
train.train()
