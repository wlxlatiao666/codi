"""Prepare immutable final checkpoints for private HF backup without touching training."""
import hashlib,json,os,shutil,tarfile
from pathlib import Path
R=Path('/home/tiger/codi-ab-c71db6e-20260909'); REPO=R/'repository'; OUT=R/'hf-export-20261023'
SPECS=[('codi-base-gpt2','attention-aligned-20261012/runs/codi','ood-aligned-20261014/results/codi','bb10ac1fc738c1779c8a27a7b581f52d371320d383db398d8253b2665db72276'),('codi-fixed-three-cut-gpt2','attention-aligned-20261012/runs/continuation','ood-aligned-20261014/results/continuation','898aafd1f269513011e95c14767ec9d1ebd535b8b3e9aa9756db0cc243af5421'),('codi-all-cuts-gpt2','all-cuts-20261020/runs/full','all-cuts-20261020/final','4f0db42fcd13ccf949ddf01b0c242a00db35545c814bbd11c401c027e74d357d')]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
for name,run,ood,expected in SPECS:
 cp=R/run/'run/gpt2/ep_40/lr_0.003/seed_11/checkpoint-119840'; dest=OUT/name;dest.mkdir(parents=True,exist_ok=True)
 assert sha(cp/'pytorch_model.bin')==expected
 for f in cp.iterdir():
  if not f.is_file():continue
  q=dest/f.name
  if q.exists():assert sha(q)==sha(f)
  else:os.link(f,q)
 (dest/'evaluation').mkdir(exist_ok=True)
 for split in ['validation','test','gsm-hard','multi-arith-full','svamp']:
  if name=='codi-all-cuts-gpt2':p=R/ood/split/'metrics.json'
  elif split=='validation':p=R/run/'validation/epoch-40/metrics.json'
  elif split=='test':p=R/run/'test/metrics.json'
  else:p=R/ood/split/'metrics.json'
  assert p.exists(),p
  shutil.copy2(p,dest/'evaluation'/f'{split}.json')
 shutil.copy2(REPO/'configs/reproduction.json',dest/'reproduction.json')
 with tarfile.open(dest/'source.tar.gz','w:gz') as tar:
  for rel in ['train.py','src/model.py','src/continuation.py','src/search_data.py','src/all_cuts.py','requirements.txt','configs/reproduction.json','scripts/reproduce/evaluate.py','scripts/reproduce/evaluate_ood.py','scripts/reproduce/eval_routing.py']:
   tar.add(REPO/rel,arcname=rel)
 readme=f'''---
tags:
- codi
- checkpoint
- gpt2
base_model: openai-community/gpt2
---
# {name}

Final epoch40 / step119840 checkpoint, seed11. Private backup including optimizer,
scheduler and RNG states. This is a custom CODI state dict, not a standalone
AutoModelForCausalLM checkpoint. Use the matching custom CODI code and GPT-2 base
assets; restore the tokenizer, LoRA r128/alpha32 and projector configuration.

Code: https://github.com/wlxlatiao666/codi/tree/27e9405a8c0576e0aa431618783ada6beddaf517
The matching core and evaluation source is also preserved in source.tar.gz.
Training dependencies: Python3.11 and pinned requirements.txt in that archive.

Source run: {run}
Fixed-three-cut means the original attention-aligned run, NOT fixed-control.
Its pairs are (3,1/3), (4,1/3), (5,2/3), auxiliary weight0.3.
All-cuts means k1..6 x all proper explicit suffixes, auxiliary weight0.3.
Base CODI has no continuation supervision.

Evaluation JSON files retain dataset hashes and inference settings. MultiArith
uses the 600-item full dataset, with source split results recorded separately.
Full training recovery additionally requires the exact external dataset caches
and matching training entry point documented in the code repository.
'''
 (dest/'README.md').write_text(readme)
 files={str(p.relative_to(dest)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in dest.rglob('*') if p.is_file() and p.name!='manifest.json'}
 manifest=dict(repo_id='wlxlatiao/'+name,private=True,checkpoint_source=str(cp),epoch=40,step=119840,files=files,total_bytes=sum(v['bytes'] for v in files.values()))
 (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps(dict(repo=manifest['repo_id'],files=len(files)+1,bytes=manifest['total_bytes'],weights_sha256=expected)))
