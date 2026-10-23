"""Audit full suffixes starting at every retained step, including step one."""
import json
import os
import sys
from pathlib import Path
from collections import Counter
import torch
from transformers import AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'local_search'))
from runtime import ROOT,REPO,BANK,sha,main_hash,write_json
sys.path.insert(0,str(REPO))
from src.continuation import split_reasoning_steps
from src.window_suffix import window_starts
exp=Path(os.environ['WINDOW_SUFFIX_ROOT']);exp.mkdir(parents=True,exist_ok=True)
previous=json.loads((ROOT/'random-disturb-20261015/data-audit.json').read_text())
aligned=json.loads((ROOT/'aligned-step-20261021/data-audit.json').read_text())
assert sha(BANK)==previous['bank_sha256']
if (exp/'data-audit.json').exists():
    audit=json.loads((exp/'data-audit.json').read_text())
    assert audit['source_bank_sha256']==previous['bank_sha256']
    for name in ['dataset','smoke']:
        assert sha(exp/('dataset.pt' if name=='dataset' else 'smoke-bank.pt'))==audit[name+'_sha256']
    print('Verified existing window suffix dataset',flush=True);sys.exit(0)
data=torch.load(BANK,map_location='cpu',weights_only=False)
assert main_hash(data)==aligned['main_data_sha256']
tok=AutoTokenizer.from_pretrained(str(ROOT/'assets/models/gpt2'),use_fast=False)
rows=[];hist=Counter();eligible=[];empty=[];examples=[];maxlen=0
for i,old in enumerate(data['dual_options']):
    enc=len(data['encoder_input_ids'][i]);alen=len(data['decoder_input_ids'][i])-1
    cot=tok.decode(data['ref_input_ids'][i][enc-1:-alen],clean_up_tokenization_spaces=False)
    steps=split_reasoning_steps(cot);m=len(steps);hist[m]+=1
    if not m:
        assert len(old)==1 and len(old[0])==1
        rows.append([]);empty.append(i);continue
    answer=tok.decode(data['decoder_input_ids'][i][1:-1],clean_up_tokenization_spaces=False)
    assert answer.startswith('The answer is:')
    body='\n'.join(steps)+'\n'+answer
    eot=int(data['decoder_input_ids'][i][0]);assert eot==50259
    full=torch.tensor([eot]+tok.encode(body,add_special_tokens=False)+[tok.eos_token_id])
    assert tok.decode(full[1:-1],clean_up_tokenization_spaces=False)==body
    if m==1:assert len(old)==1 and len(old[0])==1
    else:assert len(old)==m-1 and len(old[0])>1
    suffixes=[full]+(old if m>1 else [])
    assert len(suffixes)==m
    for target in suffixes:
        assert target[0]==eot and target[-1]==tok.eos_token_id and len(target)<=256
        maxlen=max(maxlen,len(target))
    rows.append(suffixes);eligible.append(i)
    if m in [1,2,6] and not any(x['m']==m for x in examples):
        examples.append(dict(row=i,m=m,steps=steps,windows=[window_starts(k,m) for k in range(1,7)],full_target=body))
    if i%50000==0:print('WINDOW_AUDIT',i,flush=True)
assert {str(k):v for k,v in hist.items()}==aligned['step_histogram']
data.pop('dual_options');data.pop('dual_row_id');data['window_suffixes']=rows
assert main_hash(data)==aligned['main_data_sha256']
maxenc=max(map(len,data['encoder_input_ids']));assert maxenc+6+maxlen<=512
indices=sorted(eligible[:192]+empty[:64]);assert len(indices)==256
smoke={k:[v[i] for i in indices] for k,v in data.items()}
torch.save(data,exp/'dataset.pt');torch.save(smoke,exp/'smoke-bank.pt')
write_json(exp/'data-audit.json',dict(source_bank_sha256=previous['bank_sha256'],dataset_sha256=sha(exp/'dataset.pt'),smoke_sha256=sha(exp/'smoke-bank.pt'),main_data_sha256=main_hash(data),samples=len(rows),eligible=len(eligible),empty=len(empty),step_histogram=dict(hist),examples=examples,smoke_indices=indices,max_target=maxlen,worst_padded_context=maxenc+6+maxlen,auxiliary_branches_per_epoch=6*sum(min(3,m)*n for m,n in hist.items()),target='EOT + steps j..m + answer + EOS; j is one-based first predicted step',mapping='window width min(3,m); center ceil(k*m/6); shift whole window at boundaries',aggregation='0.3 mean_k mean_valid_sample mean_candidate mean_token CE',old_proper_suffixes_reused_exactly=True))
print('WINDOW_DATA_COMPLETE',dict(hist),flush=True)
