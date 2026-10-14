"""Independent cutpoints, common eligibility, and reproducible one-branch sampling."""
import json, math, logging
from collections import Counter
from pathlib import Path
import torch
from .continuation import build_continuation, collate_continuations, split_reasoning_steps

# A common eligibility gate across the search family removes coverage confounding.
GATE_PAIRS = [(2, 1/3), (3, 1/3), (3, .5), (4, 1/3), (4, .5), (4, 2/3), (5, 2/3)]

def parse_pairs(args, checkpoint):
    pairs = json.loads(args.continuation_pairs) if args.continuation_pairs else [[checkpoint, args.continuation_cut_ratio]]
    assert pairs and len({int(k) for k,r in pairs}) == len(pairs), "Each k must map to exactly one target cut"
    for k,r in pairs:
        assert int(k) == k and 1 <= k < args.num_latent and 0 < r < 1
    return [(int(k), float(r)) for k,r in pairs]

def build_options(questions, cots, answers, tokenizer, eot, main, model, args):
    pairs = parse_pairs(args, model.continuation_checkpoint)
    config = model.codi.config
    limit = min(args.model_max_length, getattr(config, 'max_position_embeddings', getattr(config,'n_positions',args.model_max_length)))
    rows=[]; counts=Counter(); hist=Counter(); lengths=Counter()
    for idx,(q,c,a) in enumerate(zip(questions,cots,answers)):
        steps=split_reasoning_steps(c)
        segment_long = any(len(tokenizer.encode(x)) > 256 for x in (q,c,a))
        built={}
        reason='eligible'
        for k,r in set(GATE_PAIRS+pairs):
            if r not in built:
                built[r]=build_continuation(c,a,tokenizer,eot,args.remove_eos,r,args.continuation_max_tokens)
            ex,why=built[r]
            if why != 'eligible': reason=why
            elif len(main['encoder_input_ids'][idx])+k+len(ex['continuation_input_ids']) > limit:
                reason='context_too_long'
        if segment_long: reason='baseline_segment_truncated'
        counts[reason]+=1; opts=[]
        for k,r in pairs:
            ex,_=built[r]
            if reason!='eligible':
                ex={'continuation_input_ids':torch.tensor([eot]),'continuation_labels':torch.tensor([-100])}
            else:
                j=min(len(steps)-1,max(1,math.ceil(len(steps)*r)))
                hist[f'k={k},m={len(steps)},j={j}']+=1
                lengths[f'{k}:{r}']+=len(ex['continuation_labels'])-1
            ex=dict(ex)
            mask=torch.zeros_like(ex['continuation_labels'],dtype=torch.bool)
            if reason=='eligible':
                markers=1+(not args.remove_eos)
                first=tokenizer.encode(steps[j],add_special_tokens=False)
                body=ex['continuation_input_ids'][markers:].tolist()
                common=0
                for x,y in zip(first,body):
                    if x!=y: break
                    common+=1
                assert common > 0
                mask[markers:markers+common]=True
            ex['continuation_first_step_mask']=mask
            ex['continuation_checkpoint_override']=k
            opts.append(ex)
        rows.append(opts)
    assert counts['eligible'] > 0, counts
    report=dict(auxiliary_full_cot=getattr(args,"continuation_use_full_cot",False),teacher_include_last_cot=getattr(args,"include_last_cot",False),pairs=pairs,common_eligibility_gate=GATE_PAIRS,coverage=dict(counts),cut_histogram=dict(hist),supervised_tokens=dict(lengths))
    Path(args.output_dir).mkdir(parents=True,exist_ok=True)
    (Path(args.output_dir)/'coverage.json').write_text(json.dumps(report,indent=2))
    logging.warning('SEARCH_COVERAGE %s',dict(counts))
    return rows

def collate_options(instances,pad_token_id):
    options=[]
    for i in range(len(instances[0]['continuation_options'])):
        rows=[x['continuation_options'][i] for x in instances]
        out=collate_continuations(rows,pad_token_id)
        out['continuation_first_step_mask']=torch.nn.utils.rnn.pad_sequence([x['continuation_first_step_mask'] for x in rows],batch_first=True,padding_value=False)
        out['continuation_checkpoint_override']=rows[0]['continuation_checkpoint_override']
        options.append(out)
    return options
