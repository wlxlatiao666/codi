import os,sys,json,hashlib,importlib.util,copy,gc,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
torch.use_deterministic_algorithms(True)
from transformers import set_seed
from peft import LoraConfig,TaskType
REPO=Path(__file__).resolve().parents[2]
ROOT=Path(os.environ.get('CODI_ASSET_ROOT','/home/tiger/codi-ab-c71db6e-20260909'));EXP=ROOT/'attention-aligned-20261012'
sys.path.insert(0,str(REPO))
from src import model as common
from src.search_data import collate_options
spec=importlib.util.spec_from_file_location('src.historical_model',EXP/'source/src/model.py')
official=importlib.util.module_from_spec(spec);sys.modules[spec.name]=official;spec.loader.exec_module(official)
if not (EXP/'smoke-dataset.pt').exists():
 print('Loading exact previous random-cut cache',flush=True)
 d=torch.load(ROOT/'continuation-search-20260930/runs/random-cut/dataset.pt',map_location='cpu',weights_only=False)
 assert len(d['encoder_input_ids'])==383368
 assert [o['continuation_checkpoint_override'] for o in d['continuation_options'][0]]==[3,4,5]
 torch.save({k:v[:256] for k,v in d.items()},EXP/'smoke-dataset.pt')
 del d;gc.collect()
d=torch.load(EXP/'smoke-dataset.pt',weights_only=False)
inds=[i for i,x in enumerate(d['continuation_options']) if (x[0]['continuation_labels']!=-100).any() and len(d['ref_input_ids'][i])<120][:4]
assert len(inds)==4
rows=[{k:v[i] for k,v in d.items()} for i in inds]
pad=50257
batch={}
for k in ['encoder_input_ids','decoder_input_ids','ref_input_ids','labels','ref_labels']:
 vals=[x[k] for x in rows]
 if k=='encoder_input_ids': batch[k]=torch.nn.utils.rnn.pad_sequence([v.flip(0) for v in vals],batch_first=True,padding_value=pad).flip(1).cuda()
 else:batch[k]=torch.nn.utils.rnn.pad_sequence(vals,batch_first=True,padding_value=-100 if k in ['labels','ref_labels'] else pad).cuda()
for k in ['ref_answer_position','model_answer_position']:batch[k]=torch.tensor([x[k] for x in rows]).cuda()
batch['encoder_attention_mask']=batch['encoder_input_ids'].ne(pad);batch['ref_attention_mask']=batch['ref_input_ids'].ne(pad)
batch.update(step=0,step_ratio=0.)
opts=collate_options(rows,pad)
opts=[{k:v.cuda() if torch.is_tensor(v) else v for k,v in opt.items()} for opt in opts]
def build(mod,sim=False,weight=0):
 set_seed(11)
 ma=mod.ModelArguments(model_name_or_path=str(ROOT/'assets/models/gpt2'),lora_init=True,lora_r=128,lora_alpha=32,use_decoder=sim)
 ta=mod.TrainingArguments(output_dir=str(EXP/'verify'),bf16=True,num_latent=6,use_prj=True,prj_dim=768,prj_dropout=0,fix_attn_mask=False,remove_eos=True,distill_loss_div_std=True,print_loss=False,report_to='none')
 ta.continuation_weight=weight
 lc=LoraConfig(task_type=TaskType.CAUSAL_LM,inference_mode=False,r=128,lora_alpha=32,lora_dropout=.1,target_modules=['c_attn','c_proj','c_fc'],init_lora_weights=True)
 return mod.CODI(ma,ta,lc).cuda().train()
def fwd(m,extra=None):
 m.zero_grad(set_to_none=True);set_seed(123)
 with torch.autocast('cuda',dtype=torch.bfloat16):out=m(**batch,**(extra or {}))
 out['loss'].backward()
 assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
 return {k:v.detach().cpu() for k,v in out.items() if torch.is_tensor(v)}, {n:p.grad.detach().cpu().clone() for n,p in m.named_parameters() if p.grad is not None}

# Train-mode dropout, exact loss/logits, every gradient, and one AdamW update.
def tensor_hash(items):
 h=hashlib.sha256()
 for n,t in items:
  h.update(n.encode());h.update(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
def probe(mod,sim,weight,extra):
 m=build(mod,sim,weight)
 initial=tensor_hash(m.state_dict().items())
 optimizer=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.003,weight_decay=.1)
 out,grads=fwd(m,extra)
 rng=torch.cuda.get_rng_state().cpu().clone()
 torch.nn.utils.clip_grad_norm_(m.parameters(),2)
 optimizer.step()
 after={n:p.detach().cpu().clone() for n,p in m.named_parameters() if p.requires_grad}
 del m,optimizer;gc.collect();torch.cuda.empty_cache()
 return initial,out,grads,rng,after
report={}
for name,sim,weight,extra in [('codi',False,0,None),('sim',True,0,None)]+[(f'continuation-k{o["continuation_checkpoint_override"]}',False,.3,o) for o in opts]:
 a=probe(official,sim,weight,extra);b=probe(common,sim,weight,extra)
 assert a[0]==b[0],(name,'initial weights differ')
 assert a[1].keys()==b[1].keys() and a[2].keys()==b[2].keys()
 exact={}
 differences={}
 for group in [1,2,4]:
  exact[group]=all(torch.equal(a[group][key],b[group][key]) for key in a[group])
  differences[group]=max((a[group][key]-b[group][key]).abs().max().item() for key in a[group])
 print('DIAGNOSTIC',name,exact,differences,flush=True)
 for group in [1,2,4]:
  for key in a[group]:torch.testing.assert_close(a[group][key],b[group][key],rtol=0,atol=0)
 assert torch.equal(a[3],b[3])
 report[name]=dict(deterministic_algorithms_for_verification_only=True,initial_weights_equal=True,forward_and_gradients_bitwise_equal=True,rng_state_equal=True,adamw_update_equal=True,gradient_tensors=len(a[2]))
 print(name,report[name],flush=True)
 del a,b;gc.collect()
dest=Path(os.environ.get('CODI_VERIFY_OUTPUT',str(ROOT/'reorganization-checks-20261014')))
dest.mkdir(parents=True,exist_ok=True)
(dest/'model-parity.json').write_text(json.dumps(report,indent=2))
