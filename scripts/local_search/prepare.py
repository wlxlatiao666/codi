"""Read-only audit of historical data, plus a small isolated smoke cache."""
import json
import sys
from collections import Counter
import torch
from runtime import REPO, ROOT, EXP, BANK, LEGACY, sha, main_hash, write_json, OptionRows
from policy import INITIAL, cut

sys.path.insert(0, str(REPO))
from src.continuation import split_reasoning_steps
from transformers import AutoTokenizer


def main():
    EXP.mkdir(parents=True, exist_ok=True)
    historical = torch.load(LEGACY, map_location='cpu', weights_only=False)
    bank = torch.load(BANK, map_location='cpu', weights_only=False)
    ref = json.loads((REPO / 'configs/reproduction.json').read_text())
    assert sha(LEGACY) == ref['cache_sha256']['continuation']
    assert main_hash(bank) == main_hash(historical) == ref['training_reference']['main_data_sha256']
    keys = set(historical) - {'continuation_options'}
    assert set(bank) - {'dual_options', 'dual_row_id'} == keys
    for key in keys:
        assert len(bank[key]) == len(historical[key]) == 383368
        for a, b in zip(bank[key], historical[key]):
            assert torch.equal(a, b) if torch.is_tensor(a) else a == b, key
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / 'assets/models/gpt2'), use_fast=False)
    old_options = historical['continuation_options']
    mapped = OptionRows(bank['dual_options'], INITIAL)
    hist = Counter()
    eligible_indices, ineligible_indices = [], []
    target_count = 0
    for i, suffixes in enumerate(bank['dual_options']):
        allowed = len(suffixes[0]) > 1
        assert allowed == bool(old_options[i][0]['continuation_labels'].ne(-100).any())
        for old, new in zip(old_options[i], mapped[i]):
            for key in ['continuation_input_ids', 'continuation_labels']:
                assert torch.equal(old[key], new[key]), (i, key)
            assert old['continuation_checkpoint_override'] == new['continuation_checkpoint_override']
        if not allowed:
            assert len(suffixes) == 1 and suffixes[0].tolist() == [int(bank['decoder_input_ids'][i][0])]
            ineligible_indices.append(i)
            continue
        eligible_indices.append(i)
        enc_len = len(bank['encoder_input_ids'][i])
        alen = len(bank['decoder_input_ids'][i]) - 1
        cot = tokenizer.decode(bank['ref_input_ids'][i][enc_len - 1:-alen], clean_up_tokenization_spaces=False)
        answer = tokenizer.decode(bank['decoder_input_ids'][i][1:-1], clean_up_tokenization_spaces=False)
        steps = split_reasoning_steps(cot)
        assert len(steps) == len(suffixes) + 1
        hist[len(steps)] += 1
        for j, ids in enumerate(suffixes, start=1):
            expected = [int(bank['decoder_input_ids'][i][0])] + tokenizer.encode(
                '\n'.join(steps[j:]) + '\n' + answer, add_special_tokens=False) + [tokenizer.eos_token_id]
            assert ids.tolist() == expected, (i, j, 'suffix reconstruction')
            assert len(ids) <= 256 and enc_len + 5 + len(ids) <= 512
            target_count += 1
        if i % 50000 == 0:
            print('AUDIT_ROWS', i, flush=True)
    assert len(eligible_indices) == 177390
    smoke_indices = sorted(eligible_indices[:192] + ineligible_indices[:64])
    smoke = {k: [v[i] for i in smoke_indices] for k, v in bank.items()}
    torch.save(smoke, EXP / 'smoke-bank.pt')
    # Same selected rows in old format allow exact input parity checks.
    torch.save({k: [v[i] for i in smoke_indices] for k, v in historical.items()}, EXP / 'smoke-legacy.pt')
    report = dict(samples=383368, eligible=177390, all_main_fields_equal=True,
                  legacy_targets_equal=True, all_suffixes_reconstructed=True,
                  suffix_target_count=target_count, step_histogram=dict(hist),
                  all_k_1_to_5_all_j_context_safe=True, main_data_sha256=main_hash(bank),
                  bank_sha256=sha(BANK), legacy_sha256=sha(LEGACY),
                  smoke_indices=smoke_indices, ratio_denominator=6,
                  core_sha256={p: sha(REPO / p) for p in ref['core_sha256']})
    assert report['core_sha256'] == ref['core_sha256']
    write_json(EXP / 'data-audit.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'smoke_indices'}), flush=True)


if __name__ == '__main__':
    main()
