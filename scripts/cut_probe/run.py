"""Frozen CODI probe of every distinct latent/suffix pairing."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
from collections import defaultdict, Counter
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from peft import LoraConfig, TaskType
from transformers import set_seed
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.model import CODI, ModelArguments, TrainingArguments
ROOT = Path(os.environ.get('CODI_ASSET_ROOT','/home/tiger/codi-ab-c71db6e-20260909'))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def save(path, obj):
    p=Path(path); tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps(obj,indent=2)); tmp.replace(p)

def target_counts(ids, prompt, newline):
    values=ids.tolist()
    starts=[i for i in range(1,len(values)-len(prompt)+1) if values[i:i+len(prompt)]==prompt]
    assert len(starts)==1, values
    a=starts[0]
    reasoning=values[1:a]
    first=reasoning.index(newline)+1 if newline in reasoning else len(reasoning)
    return dict(total=len(values)-1, reason=a-1, answer=len(values)-a,
                first_step=first, first8=min(8,a-1), firsttoken=1)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True)
    ap.add_argument('--per-length',type=int,default=512); ap.add_argument('--batch',type=int,default=64)
    ap.add_argument('--model',choices=['base','fixed','all-cuts'],default='base')
    args=ap.parse_args(); dest=Path(args.output); dest.mkdir(parents=True,exist_ok=True)
    save(dest/'status.json',dict(stage='loading'))
    origin=ROOT/{'base':'attention-aligned-20261012/runs/codi', 'fixed':'random-disturb-20261015/runs/fixed-control', 'all-cuts':'all-cuts-20261020/runs/full'}[args.model]
    last=json.loads((origin/'last-checkpoint.json').read_text())
    assert last['epoch']==40 and last.get('global_step',last.get('step'))==119840
    if args.model=='base':
        launch=json.loads((origin/'phase-40-launch.json').read_text())['command']
        assert float(launch[launch.index('--continuation_weight')+1])==0
        provenance={'continuation_weight':0}
    elif args.model=='all-cuts':
        assert last['pairs_sixths']=='all k=1..6 x all proper suffixes'
        provenance={'continuation_weight':.3,'selection':'all-cuts final epoch40',
                    'evaluation':json.loads((ROOT/'all-cuts-20261020/results.json').read_text())}
    else:
        phase=json.loads((origin/'phase-config.json').read_text())
        assert last['pairs_sixths']==phase['pairs_sixths']==[[3,2],[4,2],[5,4]]
        provenance={'continuation_weight':.3,'phase_config':phase,
                    'evaluation':json.loads((origin/'latest-evaluation.json').read_text()),
                    'selection':'historical fixed-control epoch40; tied highest recorded validation accuracy 298/512'}
    cp=Path(last['checkpoint']); set_seed(11)
    ma=ModelArguments(model_name_or_path=str(ROOT/'assets/models/gpt2'),train=False)
    ta=TrainingArguments(output_dir=str(dest),bf16=True,use_lora=True,num_latent=6,use_prj=True,
                         prj_dim=768,fix_attn_mask=False,remove_eos=True,report_to='none')
    lora=LoraConfig(task_type=TaskType.CAUSAL_LM,inference_mode=False,r=128,lora_alpha=32,
                    lora_dropout=.1,target_modules=['c_attn','c_proj','c_fc'],init_lora_weights=True)
    model=CODI(ma,ta,lora)
    state=torch.load(cp/'pytorch_model.bin',map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True); del state
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model=model.to(device='cuda',dtype=torch.float32).eval().requires_grad_(False)
    tok=model.tokenizer
    bank_path=ROOT/'continuation-dual-random-20261010/dataset.pt'
    previous=json.loads((ROOT/'random-disturb-20261015/data-audit.json').read_text())
    assert sha(bank_path)==previous['bank_sha256']
    bank=torch.load(bank_path,map_location='cpu',weights_only=False)
    groups=defaultdict(list)
    for i,suffixes in enumerate(bank['dual_options']):
        if len(suffixes[0])>1: groups[len(suffixes)+1].append(i)
    rng=random.Random(20261020); rows=[]; counts={}
    for m,indices in sorted(groups.items()):
        selected=sorted(rng.sample(indices,min(args.per_length,len(indices))))
        counts[m]=dict(population=len(indices),sample=len(selected))
        for i in selected:
            rows.append(dict(dataset='train_stratified',row=i,m=m,weight=len(indices)/len(selected),
                             encoder=bank['encoder_input_ids'][i],suffixes=bank['dual_options'][i]))
    del bank
    validation=load_from_disk(str(ROOT/'continuation-search-20260930/data/validation'))['validation']
    excluded=Counter()
    for i,row in enumerate(validation):
        steps=re.findall(r'<<[^<>]+>>',row['answer'])[:-1]
        if len(steps)<2: excluded['fewer_than_two_retained_equations']+=1; continue
        answer='The answer is: '+row['answer'].split('####')[-1].strip().replace(',','')
        enc=tok.encode(row['question'].strip().replace('  ',' '),add_special_tokens=False)+[model.bot_id]
        suffixes=[torch.tensor([model.eot_id]+tok.encode('\n'.join(steps[j:])+'\n'+answer,
                    add_special_tokens=False)+[tok.eos_token_id]) for j in range(1,len(steps))]
        if len(enc)>257 or max(map(len,suffixes))>256:
            excluded['too_long']+=1; continue
        rows.append(dict(dataset='validation_equations',row=i,m=len(steps),weight=1,
                         encoder=torch.tensor(enc),suffixes=suffixes))
    # Match physical padding within each fixed batch across every k and every j.
    # Shuffle training rows to avoid sorting all batches by chain length.
    training=[r for r in rows if r['dataset']=='train_stratified']; rng.shuffle(training)
    rows=training+[r for r in rows if r['dataset']=='validation_equations']
    prompt=tok.encode('The answer is:',add_special_tokens=False)
    newline=tok.encode('\n',add_special_tokens=False); assert len(newline)==1
    meta=dict(checkpoint=str(cp),checkpoint_sha256=sha(cp/'pytorch_model.bin'),source=last,
              model_label={'base':'Base CODI','fixed':'Fixed-setting CODI','all-cuts':'All-cuts CODI'}[args.model],
              training_provenance=provenance,base_continuation_weight=provenance["continuation_weight"],model_eval=True,requires_grad=False,k_values=list(range(7)),
              train_strata=counts,train_population=sum(len(v) for v in groups.values()),
              datasets=dict(Counter(r['dataset'] for r in rows)),validation_excluded=dict(excluded),
              validation_preprocessing='extract <<equations>> from gold answer; remove final equation; match training suffix syntax',
              train_sampling='up to 512 per retained-step-count m; seed20261020; weight=population/sample',
              targets='all proper suffixes j=1..m-1; EOT + suffix + answer + EOS',
              r_mapping='j=min(m-1,max(1,ceil(m*r))); every distinct j evaluated exactly once',
              precision='float32 forward and CE; TF32 disabled; natural logarithm nats/token',batch=args.batch,
              prefix_attention='historical left padding; encoder mask only; implicit positions',
              bank_sha256=previous['bank_sha256'])
    save(dest/'manifest.json',meta)
    fields=['dataset','row','m','j','r','weight','k']+[x+'_'+y for x in ['total','reason','answer','first_step','first8','firsttoken'] for y in ['sum','n']]
    started=time.time(); evaluated=0; checks=0
    with (dest/'per_target.csv').open('w') as f, torch.inference_mode():
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for dataset in ['train_stratified','validation_equations']:
            subset=[r for r in rows if r['dataset']==dataset]
            for start in range(0,len(subset),args.batch):
                batch=subset[start:start+args.batch]; width=max(len(r['encoder']) for r in batch)
                ids=torch.full((len(batch),width),model.pad_token_id,dtype=torch.long,device='cuda')
                mask=torch.zeros_like(ids)
                branches=[]
                for owner,row in enumerate(batch):
                    ids[owner,-len(row['encoder']):]=row['encoder'].to('cuda'); mask[owner,-len(row['encoder']):]=1
                    for j,target in enumerate(row['suffixes'],1):
                        branches.append((owner,row,j,target,target_counts(target,prompt,newline[0])))
                out=model.codi(input_ids=ids,attention_mask=mask,use_cache=True,output_hidden_states=True)
                cache=out.past_key_values; latent=model.prj(out.hidden_states[-1][:,-1:])
                for k in range(7):
                    if k:
                        out=model.codi(inputs_embeds=latent,past_key_values=cache,use_cache=True,output_hidden_states=True)
                        cache=out.past_key_values; latent=model.prj(out.hidden_states[-1][:,-1:])
                    before=cache[0][0].shape[-2]
                    assert before==width+k
                    for offset in range(0,len(branches),32):
                        block=branches[offset:offset+32]
                        targets=torch.nn.utils.rnn.pad_sequence([b[3] for b in block],batch_first=True,padding_value=model.pad_token_id).to('cuda')
                        labels=torch.nn.utils.rnn.pad_sequence([b[3] for b in block],batch_first=True,padding_value=-100).to('cuda'); labels[:,0]=-100
                        owners=torch.tensor([b[0] for b in block],device='cuda')
                        copied=tuple(tuple(t.index_select(0,owners) for t in layer) for layer in cache)
                        assert width+k+targets.shape[1]<=1024
                        logits=model.codi(input_ids=targets,past_key_values=copied,use_cache=True).logits[:,:-1].float()
                        loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),labels[:,1:].reshape(-1),ignore_index=-100,reduction='none').reshape(len(block),-1)
                        assert torch.isfinite(loss).all()
                        # Independently verify one branch at every k and both datasets.
                        if start==0 and offset==0:
                            single=tuple(tuple(t[0:1].clone() for t in layer) for layer in copied)
                            one=model.codi(input_ids=targets[:1],past_key_values=single,use_cache=True).logits[:,:-1].float()
                            reference=F.cross_entropy(one.reshape(-1,one.size(-1)),labels[:1,1:].reshape(-1),ignore_index=-100,reduction='sum')
                            torch.testing.assert_close(loss[0].sum(),reference,atol=.002,rtol=.00002)
                            checks+=1
                        loss=loss.cpu()
                        for bi,(_,row,j,_,ns) in enumerate(block):
                            record={key:row[key] for key in ['dataset','row','m','weight']}; record.update(j=j,r=j/row['m'],k=k)
                            for name,n in ns.items():
                                segment=loss[bi,ns['reason']:ns['total']] if name=='answer' else loss[bi,:n]
                                record[name+'_sum']=float(segment.sum()); record[name+'_n']=n
                            assert abs(record['total_sum']-record['reason_sum']-record['answer_sum'])<.002
                            writer.writerow(record); evaluated+=1
                        assert cache[0][0].shape[-2]==before
                f.flush()
                save(dest/'status.json',dict(stage='running',dataset=dataset,completed=min(start+len(batch),len(subset)),
                     dataset_size=len(subset),evaluated_pairs=evaluated,elapsed_seconds=time.time()-started))
                print(json.dumps(dict(dataset=dataset,rows=min(start+len(batch),len(subset)),pairs=evaluated)),flush=True)
    meta.update(duration_seconds=time.time()-started,scored_pairs=evaluated,single_branch_checks=checks,
                peak_cuda_bytes=torch.cuda.max_memory_allocated())
    save(dest/'manifest.json',meta); save(dest/'status.json',dict(stage='complete',**{k:meta[k] for k in ['duration_seconds','scored_pairs','single_branch_checks']}))
    print('COMPLETE',json.dumps(meta),flush=True)
if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        output=Path(sys.argv[sys.argv.index('--output')+1])
        if output.exists(): save(output/'status.json',dict(stage='failed',error=repr(error)))
        raise
