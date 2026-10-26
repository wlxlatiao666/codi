"""Bound the new trainer's allocator; retain the original SIM-CoT training code."""
import os,runpy
from pathlib import Path
import torch
fraction=float(os.environ.get('SIM_TRAIN_MEMORY_FRACTION','0.8'))
assert 0 < fraction <= .8
# CUDA_VISIBLE_DEVICES selects physical GPU3; leave >=20% for serialized evals.
torch.cuda.set_per_process_memory_fraction(fraction,0)
print('SIM_TRAIN_MEMORY_FRACTION',fraction,flush=True)
runpy.run_path(str(Path(__file__).resolve().parents[1]/'reproduce/train_entry.py'),run_name='__main__')
