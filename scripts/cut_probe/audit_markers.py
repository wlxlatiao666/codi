from pathlib import Path
from collections import Counter
import csv,json,torch
root=Path('/home/tiger/codi-ab-c71db6e-20260909'); dest=root/'base-codi-cut-probe-20261020'
with (dest/'per_target.csv').open() as f:
 rows={int(r['row']) for r in csv.DictReader(f) if r['dataset']=='train_stratified'}
bank=torch.load(root/'continuation-dual-random-20261010/dataset.pt',map_location='cpu',weights_only=False)
counts=Counter(int(target[1]) for i in rows for target in bank['dual_options'][i])
vocab=json.loads((root/'assets/models/gpt2/vocab.json').read_text())
assert set(counts)=={vocab['<<']},counts
report=dict(sampled_training_rows=len(rows),target_first_token_counts=dict(counts),marker='<<',token_id=vocab['<<'],all_training_targets_start_with_marker=True,
 validation_targets_start_with_marker='guaranteed by extraction of <<...>> equations')
(dest/'marker-audit.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report))
