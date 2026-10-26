"""Archive the completed GPU0/GPU1 experiments without altering source checkpoints."""
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path

ROOT = Path('/home/tiger/codi-ab-c71db6e-20260909')
REPO = ROOT / 'repository'
OUT = ROOT / 'hf-export-20261026'
SPECS = [
    ('codi-single-cut-k3-r50-gpt2', 'single-cut-k3-r50-20261023',
     'Single continuation cut (k=3, r=1/2), predicting the suffix and answer; auxiliary weight 0.3.'),
    ('codi-aligned-step-gpt2', 'aligned-step-20261021',
     'Each latent k predicts only explicit step j=ceil(k*m/6), with no later steps or answer; weight 0.05 per latent, total 0.3.'),
]

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

for name, experiment, description in SPECS:
    run = ROOT / experiment / 'runs/full'
    last = json.loads((run / 'last-checkpoint.json').read_text())
    assert last['epoch'] == 40 and last['step'] == 119840
    cp = Path(last['checkpoint'])
    for required in ['pytorch_model.bin', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json']:
        assert (cp / required).stat().st_size > 0
    dest = OUT / name
    dest.mkdir(parents=True, exist_ok=True)
    for source in cp.iterdir():
        if source.is_file():
            target = dest / source.name
            if target.exists():
                assert sha(target) == sha(source)
            else:
                os.link(source, target)
    results = json.loads((ROOT / experiment / 'results.json').read_text())
    (dest / 'evaluation').mkdir(exist_ok=True)
    for split in ['validation', 'test', 'gsm-hard', 'multi-arith-full', 'svamp']:
        assert results[split]['checkpoint'] == str(cp)
        (dest / 'evaluation' / (split + '.json')).write_text(json.dumps(results[split], indent=2))
    shutil.copy2(REPO / 'configs/reproduction.json', dest / 'reproduction.json')
    with tarfile.open(dest / 'source.tar.gz', 'w:gz') as archive:
        for rel in ['train.py', 'requirements.txt', 'src', 'scripts', 'configs']:
            archive.add(REPO / rel, arcname=rel,
                        filter=lambda info: None if '__pycache__' in info.name or info.name.endswith('.pyc') else info)
    (dest / 'README.md').write_text(f'''---
tags:
- codi
- gpt2
- checkpoint
base_model: openai-community/gpt2
---
# {name}

{description}

Final epoch 40 / step 119840, seed 11. GPT-2 with LoRA r128/alpha32,
6 latent positions, batch64 with accumulation2, learning rate0.003.
This private archive includes the full custom CODI state dict (base weights,
LoRA weights and projector), tokenizer, optimizer, scheduler and RNG state.
It is not a merged standalone AutoModelForCausalLM export.

Use the matching custom code in source.tar.gz and GPT-2 base assets to load it.
Full training recovery also requires the original external dataset caches.
Source experiment: {experiment}
Code project: https://github.com/wlxlatiao666/codi
The source archive includes the experiment-specific launchers used for this run.

Evaluation records include dataset hashes and decoding settings. MultiArith
uses all 600 examples, including both source train and test splits.
''')
    files = {str(p.relative_to(dest)): dict(bytes=p.stat().st_size, sha256=sha(p))
             for p in dest.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    manifest = dict(repo_id='wlxlatiao/' + name, private=True,
                    checkpoint_source=str(cp), epoch=40, step=119840,
                    files=files, total_bytes=sum(v['bytes'] for v in files.values()))
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(repo=manifest['repo_id'], files=len(files)+1,
                          bytes=manifest['total_bytes'])), flush=True)
