"""Reuse the exhaustive suffix reconstruction audit; verify new k=6 bounds."""
import json
import os
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'local_search'))
from runtime import ROOT, REPO, BANK, sha, main_hash, write_json
exp = Path(os.environ['ALL_CUTS_ROOT'])
previous = json.loads((ROOT / 'random-disturb-20261015/data-audit.json').read_text())
assert previous['all_suffixes_reconstructed'] and previous['all_main_fields_equal']
assert sha(BANK) == previous['bank_sha256']
data = torch.load(BANK, map_location='cpu', weights_only=False)
assert main_hash(data) == previous['main_data_sha256']
max_encoder = max(map(len, data['encoder_input_ids']))
max_suffix = max(len(t) for row in data['dual_options'] for t in row)
max_positions = json.loads((ROOT / 'assets/models/gpt2/config.json').read_text())['n_positions']
# Actual prefix is batch-padded; test the conservative bound for ANY batch.
assert max_encoder + 6 + max_suffix <= max_positions
report = dict(worker=4423662, bank_sha256=previous['bank_sha256'], main_data_sha256=main_hash(data),
              samples=len(data['dual_options']), eligible=previous['eligible'],
              suffix_target_count=previous['suffix_target_count'], latent_cuts=list(range(1,7)),
              branches_per_epoch=6*previous['suffix_target_count'], max_encoder=max_encoder,
              max_suffix=max_suffix, worst_padded_context=max_encoder+6+max_suffix,
              model_context_limit=max_positions, previous_reconstruction_audit=previous,
              aggregation='sum of token CE / total valid tokens over all six positions and suffixes',
              continuation_weight=.3)
write_json(exp / 'data-audit.json', report)
print(json.dumps({k:v for k,v in report.items() if k!='previous_reconstruction_audit'}), flush=True)
