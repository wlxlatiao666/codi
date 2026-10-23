"""Reconstruct all retained explicit steps, with no suffix or answer targets."""
import json
import os
import sys
from pathlib import Path
from collections import Counter
import torch
from transformers import AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'local_search'))
from runtime import ROOT, REPO, BANK, sha, main_hash, write_json
sys.path.insert(0, str(REPO))
from src.continuation import split_reasoning_steps
from src.aligned_step import step_index

exp=Path(os.environ['ALIGNED_STEP_ROOT']); exp.mkdir(parents=True,exist_ok=True)
ref=json.loads((REPO/'configs/reproduction.json').read_text())
previous=json.loads((ROOT/'random-disturb-20261015/data-audit.json').read_text())
assert sha(BANK)==previous['bank_sha256']
if (exp/'data-audit.json').exists():
    audit=json.loads((exp/'data-audit.json').read_text())
    assert audit['source_bank_sha256']==previous['bank_sha256']
    assert sha(exp/'dataset.pt')==audit['dataset_sha256']
    assert sha(exp/'smoke-bank.pt')==audit['smoke_sha256']
    print('Verified existing aligned-step dataset',flush=True)
    sys.exit(0)
data=torch.load(BANK,map_location='cpu',weights_only=False)
assert main_hash(data)==ref['training_reference']['main_data_sha256']
data.pop('dual_options');data.pop('dual_row_id')
tok=AutoTokenizer.from_pretrained(str(ROOT/'assets/models/gpt2'),use_fast=False)
all_steps=[];hist=Counter();max_target=0;eligible=[];empty=[];examples=[]
for i in range(len(data['encoder_input_ids'])):
    enc_len=len(data['encoder_input_ids'][i]); alen=len(data['decoder_input_ids'][i])-1
    cot=tok.decode(data['ref_input_ids'][i][enc_len-1:-alen],clean_up_tokenization_spaces=False)
    steps=split_reasoning_steps(cot)
    targets=[]
    eot=int(data['decoder_input_ids'][i][0]);assert eot==50259
    for step in steps:
        assert 'The answer is:' not in step and '####' not in step
        body=tok.encode(step,add_special_tokens=False)
        assert body and tok.eos_token_id not in body and eot not in body
        ids=torch.tensor([eot]+body+[tok.eos_token_id],dtype=torch.long)
        assert tok.decode(ids[1:-1],clean_up_tokenization_spaces=False)==step
        assert len(ids)<=256
        targets.append(ids);max_target=max(max_target,len(ids))
    hist[len(steps)]+=1
    (eligible if steps else empty).append(i)
    all_steps.append(targets)
    if len(steps) in [1,2,6] and all(e['m']!=len(steps) for e in examples):
        examples.append(dict(row=i,m=len(steps),steps=steps,mapping=[step_index(k,len(steps)) for k in range(1,7)],
            targets=[tok.decode(t[1:-1],clean_up_tokenization_spaces=False) for t in targets]))
    if i%50000==0:print('STEP_AUDIT',i,dict(hist),flush=True)
data['aligned_steps']=all_steps
assert len(all_steps)==383368
assert sum(n for m,n in hist.items() if m>=2)==previous['eligible']
assert {str(m):n for m,n in hist.items() if m>=2}==previous['step_histogram']
maxenc=max(map(len,data['encoder_input_ids']));limit=json.loads((ROOT/'assets/models/gpt2/config.json').read_text())['n_positions']
assert maxenc+6+max_target<=min(512,limit)
assert main_hash(data)==ref['training_reference']['main_data_sha256']
# Exercise empty rows too, if present; fill remainder with valid single-step tasks.
indices=sorted(eligible[:256-min(64,len(empty))]+empty[:64])
assert len(indices)==256
smoke={k:[v[i] for i in indices] for k,v in data.items()}
torch.save(data,exp/'dataset.pt');torch.save(smoke,exp/'smoke-bank.pt')
write_json(exp/'data-audit.json',dict(source_bank_sha256=previous['bank_sha256'],
    dataset_sha256=sha(exp/'dataset.pt'),smoke_sha256=sha(exp/'smoke-bank.pt'),
    main_data_sha256=main_hash(data),samples=len(all_steps),eligible=len(eligible),empty=len(empty),
    step_histogram=dict(hist),examples=examples,smoke_indices=indices,
    mapping='j=ceil(k*m/6), one based',target='EOT + exactly step j + EOS; no later step or answer',
    preprocessing='same retained CoT as baseline; final equation already removed; m=1 is eligible',
    aggregation='0.3 * mean_k(token-mean CE_k); each k coefficient=0.05',
    max_target=max_target,worst_padded_context=maxenc+6+max_target,
    auxiliary_branches_per_epoch=len(eligible)*6))
print('STEP_DATA_COMPLETE',dict(hist),flush=True)
