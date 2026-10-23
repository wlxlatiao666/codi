"""Shared paths and atomic state writes for the bounded search."""
import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('CODI_ASSET_ROOT', '/home/tiger/codi-ab-c71db6e-20260909'))
EXP = Path(os.environ.get('CODI_SEARCH_ROOT', str(ROOT / 'random-disturb-20261015')))
BANK = ROOT / 'continuation-dual-random-20261010/dataset.pt'
LEGACY = ROOT / 'continuation-search-20260930/runs/random-cut/dataset.pt'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main_hash(data):
    h = hashlib.sha256()
    for key in ['encoder_input_ids', 'decoder_input_ids', 'ref_input_ids']:
        h.update(key.encode())
        for t in data[key]:
            h.update(t.numpy().tobytes())
    return h.hexdigest()


class OptionRows:
    """Materialize only the three fixed targets needed by one minibatch row."""
    def __init__(self, suffixes, pairs):
        from policy import validate
        self.suffixes = suffixes
        self.pairs = validate(pairs)

    def __len__(self):
        return len(self.suffixes)

    def __getitem__(self, index):
        import torch
        from policy import cut
        suffixes = self.suffixes[index]
        eligible = len(suffixes[0]) > 1
        result = []
        for k, q in self.pairs:
            ids = suffixes[cut(len(suffixes) + 1, q) - 1] if eligible else suffixes[0]
            labels = ids.clone()
            labels[0] = -100
            result.append(dict(continuation_input_ids=ids, continuation_labels=labels,
                               continuation_checkpoint_override=k,
                               continuation_first_step_mask=torch.zeros_like(ids, dtype=torch.bool)))
        return result
