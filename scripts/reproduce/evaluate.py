"""Matched, offline, greedy GSM8k evaluation with historical attention and implicit physical positions."""
import os
import argparse
import json
import re
import sys
import time
from pathlib import Path
from eval_routing import route_evaluation
_eval_gpu_guard = route_evaluation()  # Keep the resource lock until process exit.
import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType
from transformers import set_seed

ROOT = Path(os.environ.get('CODI_ASSET_ROOT','/home/tiger/codi-ab-c71db6e-20260909'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.model import CODI, ModelArguments, TrainingArguments
from src.continuation import position_ids_from_mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--sim', action='store_true')
    parser.add_argument('--split', choices=['validation','test'], default='validation')
    args = parser.parse_args()
    set_seed(11)
    model_args = ModelArguments(model_name_or_path=str(ROOT / 'assets/models/gpt2'), train=False)
    training_args = TrainingArguments(output_dir=args.output, bf16=True, use_lora=True,
        num_latent=6, use_prj=True, prj_dim=768, fix_attn_mask=False, remove_eos=True,
        report_to='none')
    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=128,
        lora_alpha=32, lora_dropout=0.1, target_modules=['c_attn','c_proj','c_fc'], init_lora_weights=True)
    model = CODI(model_args, training_args, lora)
    state = torch.load(Path(args.checkpoint) / 'pytorch_model.bin', map_location='cpu', weights_only=True)
    if args.sim:
        assert any(k.startswith('decoder.') for k in state)
        state={k:v for k,v in state.items() if not k.startswith('decoder.')}
    model.load_state_dict(state, strict=True)
    del state
    model = model.to(device='cuda', dtype=torch.bfloat16).eval()
    tokenizer = model.tokenizer
    tokenizer.padding_side = 'left'
    data = (load_from_disk(str(ROOT / 'continuation-search-20260930/data/validation'))['validation'] if args.split == 'validation' else load_from_disk(str(ROOT / 'gsm8k-aug-id-20261014/data'))['test'])
    if args.limit:
        data = data.select(range(min(args.limit, len(data))))
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    started = time.time()
    correct = 0
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), (destination / 'predictions.jsonl').open('w') as log:
        for offset in range(0, len(data), 128):
            rows = data.select(range(offset, min(offset + 128, len(data))))
            batch = tokenizer([q.strip().replace('  ', ' ') for q in rows['question']],
                padding=True, return_tensors='pt').to('cuda')
            ids = torch.cat([batch.input_ids, torch.full((len(rows),1),model.bot_id,device='cuda')],1)
            mask = torch.cat([batch.attention_mask, torch.ones((len(rows),1),device='cuda',dtype=torch.long)],1)
            out = model.codi(input_ids=ids, attention_mask=mask,
                use_cache=True,output_hidden_states=True)
            cache = out.past_key_values
            latent = model.prj(out.hidden_states[-1][:,-1:].contiguous())
            for _ in range(6):
                mask = torch.cat([mask,torch.ones_like(mask[:,:1])],1)
                out = model.codi(inputs_embeds=latent,past_key_values=cache,
                    
                    use_cache=True,output_hidden_states=True)
                cache = out.past_key_values
                latent = model.prj(out.hidden_states[-1][:,-1:])
            token = torch.full((len(rows),1),model.eot_id,device='cuda',dtype=torch.long)
            finished = torch.zeros(len(rows),device='cuda',dtype=torch.bool)
            sequences = [[] for _ in rows]
            for _ in range(256):
                mask = torch.cat([mask,torch.ones_like(mask[:,:1])],1)
                out = model.codi(input_ids=token,past_key_values=cache,use_cache=True)
                cache = out.past_key_values
                next_ids = out.logits[:,-1,:model.pad_token_id].argmax(-1)
                for i, value in enumerate(next_ids.tolist()):
                    if not finished[i]:
                        sequences[i].append(value)
                finished |= next_ids.eq(tokenizer.eos_token_id)
                token = next_ids[:,None]
                if finished.all():
                    break
            for i, row in enumerate(rows):
                text = tokenizer.decode(sequences[i], skip_special_tokens=True)
                numbers = re.findall(r'-?\d+\.?\d*',text.replace(',',''))
                predicted = float(numbers[-1]) if numbers else None
                gold = float(row['answer'].split('####')[-1].replace(',','').strip())
                complete = tokenizer.eos_token_id in sequences[i]
                answer_format = 'The answer is:' in text
                ok = predicted == gold
                correct += int(ok)
                log.write(json.dumps(dict(index=offset+i,prediction=text,value=predicted,gold=gold,correct=ok,complete=complete,answer_format=answer_format))+'\n')
            log.flush()
            print(f'eval {min(offset+128,len(data))}/{len(data)} correct={correct}',flush=True)
    metrics = dict(accuracy=correct/len(data),correct=correct,total=len(data),
        duration_seconds=time.time()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        checkpoint=args.checkpoint,decoding='greedy',latent_steps=6,max_answer_tokens=256,inference_protocol="official-style padding positions and cached attention; batch128")
    predictions = [json.loads(x) for x in (destination/'predictions.jsonl').read_text().splitlines()]
    metrics.update(split=args.split, strict_accuracy=sum(x['correct'] and x['complete'] and x['answer_format'] for x in predictions)/len(predictions), truncated=sum(not x['complete'] for x in predictions))
    (destination/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics),flush=True)

if __name__ == '__main__':
    main()
