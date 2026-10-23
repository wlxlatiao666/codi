"""Bounded, restartable parent/child search, fixed control, and final evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from policy import INITIAL, propose, selection, cut
from runtime import REPO, ROOT, EXP, LEGACY, BANK, sha, write_json

SCRIPTS = REPO / 'scripts/local_search'


def environment(gpu):
    env = os.environ.copy()
    for key in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
                'SEARCH_RESUME', 'SEARCH_SMOKE']:
        env.pop(key, None)
    env.update(NO_COLOR='1', TERM='dumb', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1',
               CUDA_VISIBLE_DEVICES=str(gpu), SEARCH_EVAL_GPU='3')
    return env


def evaluate(checkpoint, destination, split='validation', smoke=False):
    destination = Path(destination)
    metric = destination / 'metrics.json'
    if metric.exists():
        return json.loads(metric.read_text())
    destination.mkdir(parents=True, exist_ok=True)
    script = 'evaluate.py' if split in {'validation', 'test'} else 'evaluate_ood.py'
    cmd = [sys.executable, str(REPO / 'scripts/reproduce' / script),
           '--checkpoint', checkpoint, '--output', str(destination), '--split', split]
    if smoke:
        cmd += ['--limit', '16']
    with (EXP / 'eval.lock').open('w') as lock, (destination / 'eval.log').open('w') as log:
        fcntl.flock(lock, fcntl.LOCK_EX)
        subprocess.run(cmd, cwd=REPO, env=environment(3), stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads(metric.read_text())


def run_phase(name, pairs, stop, gpu, resume=None, every=False, smoke=False):
    run = EXP / 'runs' / name
    run.mkdir(parents=True, exist_ok=True)
    config = dict(name=name, pairs_sixths=pairs, stop=stop, gpu=gpu,
                  source_checkpoint=resume, evaluate_each_epoch=every, smoke=smoke)
    manifest = run / 'phase-config.json'
    if manifest.exists():
        assert json.loads(manifest.read_text()) == config, ('phase configuration changed', name)
    else:
        write_json(manifest, config)
    last_file = run / 'last-checkpoint.json'
    if last_file.exists():
        last = json.loads(last_file.read_text())
        # Recover an evaluation interrupted after checkpoint save.
        if every or smoke or round(last['epoch']) % 4 == 0 or last['epoch'] >= stop:
            evaluate(last['checkpoint'], run / 'validation' / f"epoch-{last['epoch']:g}", smoke=smoke)
        if last['epoch'] >= stop - 1e-6:
            return last
        resume = last['checkpoint']
    env = environment(gpu)
    env.update(SEARCH_RUN=str(run), SEARCH_STOP=str(stop), SEARCH_PAIRS=json.dumps(pairs),
               SEARCH_EVAL_EVERY='1' if every else '0', CODI_DATA_CACHE=str(LEGACY))
    if resume:
        env['SEARCH_RESUME'] = resume
    if smoke:
        env['SEARCH_SMOKE'] = '1'
    values = dict(output_dir=str(run), expt_name='run', model_name_or_path=str(ROOT / 'assets/models/gpt2'),
        data_name='icot', seed=11, model_max_length=512, per_device_train_batch_size=64,
        gradient_accumulation_steps=2, bf16=True, num_train_epochs=40, learning_rate=.003,
        max_grad_norm=2, use_lora=True, lora_init=True, lora_r=128, lora_alpha=32,
        save_strategy='no', save_total_limit=1, save_safetensors=False, weight_decay=.1,
        warmup_ratio=.03, lr_scheduler_type='cosine', report_to='none', logging_steps=20,
        disable_tqdm=True, num_latent=6, use_prj=True, prj_dim=768, prj_dropout=0,
        distill_loss_type='smooth_l1', distill_loss_div_std=True, distill_loss_factor=1,
        ref_loss_factor=1, remove_eos=True, fix_attn_mask=False, include_last_cot=False,
        use_decoder=False, continuation_weight=.3, explain_loss_factor=1,
        print_ref_model_stats=False, print_loss=False, remove_unused_columns=False)
    if smoke:
        values.update(logging_steps=1, full_determinism=True)
    cmd = [sys.executable, str(SCRIPTS / 'train_phase.py')]
    for key, value in values.items():
        cmd += ['--' + key, str(value)]
    stamp = time.time_ns()
    write_json(run / f'launch-{stamp}.json', dict(command=cmd, resume=resume, gpu=gpu))
    with (run / 'train.log').open('a') as log:
        process = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        write_json(run / 'process.json', dict(pid=process.pid, started=time.time()))
        code = process.wait()
    write_json(run / f'exit-{stamp}.json', dict(returncode=code, ended=time.time()))
    if code:
        raise RuntimeError(f'{name} exited {code}; inspect {run}/train.log')
    last = json.loads(last_file.read_text())
    assert last['epoch'] >= stop - 1e-6
    return last


def compare_starts(names):
    reports = []
    traces = []
    for name in names:
        run = EXP / 'runs' / name
        paths = sorted(run.glob('start-*.json'), key=lambda p: int(p.stem.split('-')[1]))
        reports.append(json.loads(paths[0].read_text()))
        traces.append([json.loads(x) for x in (run / 'first-batches.jsonl').read_text().splitlines()[:2]])
    for other in reports[1:]:
        for key in ['start_step', 'max_steps', 'samples', 'main_data_sha256', 'model_state_sha256',
                    'source_checkpoint_sha256', 'scheduler_last_epoch', 'lr', 'optimizer_state_entries']:
            assert other[key] == reports[0][key], (key, names)
    assert all(trace == traces[0] for trace in traces[1:]), ('resume data order', names)
    return dict(identical_start_and_first_two_batches=True, runs=names,
                start_step=reports[0]['start_step'], model_state_sha256=reports[0]['model_state_sha256'])


def protocol():
    return dict(version=1, worker=4401371, initial_pairs_sixths=INITIAL, ratio_denominator=6,
        train_seed=11, mutation_seed=20261015, warmup_epochs=8, total_epochs=40,
        rounds=[8, 12, 16, 20], competition_epochs=2, frozen_after_epoch=24,
        mutation='one coordinate +/-1; k in 1..5, q in 1..5; ordered unique k and nondecreasing q; skip identical actual cuts',
        choice='choose a legal mutation family uniformly, then one effective neighbor uniformly',
        selection='mean validation answer accuracy at both competition endpoints; child requires >=0.01 absolute gain; otherwise parent',
        validation_samples=512, auxiliary_weight=.3, shared_warmup=True,
        control='original three pairs, same epoch8 checkpoint, continued to epoch40 without selection',
        budget='80 single-model epoch equivalents including shared warmup, control, and discarded branches; plus evaluation and smoke',
        gpu=dict(parent=0, child=1, fixed_control=2, serial_evaluation=3),
        target_cache=str(BANK), main_attention='unchanged historical physical positions; fix_attn_mask=False',
        test_policy='no test in search; final epoch40 of adaptive and fixed control only',
        evaluation_splits=['test', 'gsm-hard', 'multi-arith-full', 'svamp'],
        inference='6 latents, batch128, greedy, max256 answer tokens, historical split boundaries',
        limitations=['single training seed', 'validation reused for four selections',
                    'local two-epoch gains do not establish global optimum',
                    'adaptive policy trajectory is not proof of a superior from-scratch fixed setting'])


def smoke():
    first = run_phase('smoke/warmup', INITIAL, 1, 0, smoke=True)
    candidate = propose(INITIAL, [2, 3, 4, 5], 0)['pairs']
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(run_phase, 'smoke/' + name, pairs, 2, gpu,
                            first['checkpoint'], False, True)
                for name, pairs, gpu in [('parent', INITIAL, 0), ('identical', INITIAL, 1), ('mutant', candidate, 2)]]
        parent, identical, mutant = [j.result() for j in jobs]
    start = compare_starts(['smoke/parent', 'smoke/identical', 'smoke/mutant'])
    uninterrupted = run_phase('smoke/uninterrupted', INITIAL, 2, 0, smoke=True)
    import torch
    def load(cp, file):
        return torch.load(Path(cp) / file, map_location='cpu', weights_only=False)
    def equal(a, b):
        if torch.is_tensor(a):
            assert torch.equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                equal(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                equal(x, y)
        elif hasattr(a, 'shape'):
            import numpy as np
            assert np.array_equal(a, b)
        else:
            assert a == b
    for file in ['pytorch_model.bin', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth']:
        reference = load(parent['checkpoint'], file)
        equal(reference, load(identical['checkpoint'], file))
        equal(reference, load(uninterrupted['checkpoint'], file))
    mutant_weights = load(mutant['checkpoint'], 'pytorch_model.bin')
    parent_weights = load(parent['checkpoint'], 'pytorch_model.bin')
    assert any(not torch.equal(v, mutant_weights[k]) for k, v in parent_weights.items())
    result = dict(start, identical_arms_bitwise_equal=True, resumed_matches_uninterrupted=True,
                  model_optimizer_scheduler_rng_checked=True, mutant_changes_weights=True,
                  mutant_pairs_sixths=candidate, deterministic_smoke_only=True)
    write_json(EXP / 'smoke-verification.json', result)
    print(json.dumps(result), flush=True)


def search():
    audit = json.loads((EXP / 'data-audit.json').read_text())
    assert sha(BANK) == audit['bank_sha256']
    assert json.loads((EXP / 'smoke-verification.json').read_text())['resumed_matches_uninterrupted']
    config = protocol()
    if (EXP / 'protocol.json').exists():
        assert json.loads((EXP / 'protocol.json').read_text()) == config
    else:
        write_json(EXP / 'protocol.json', config)
    lengths = sorted(map(int, audit['step_histogram']))
    state = run_phase('warmup', INITIAL, 8, 0)
    pairs = INITIAL
    selections = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        control_job = pool.submit(run_phase, 'fixed-control', INITIAL, 40, 2, state['checkpoint'])
        for index, start in enumerate(config['rounds']):
            if state['epoch'] < start - 1e-6:
                state = run_phase(f'advance-{start}', pairs, start, 0, state['checkpoint'])
            mutation = propose(pairs, lengths, index)
            changed = sum(count for m, count in ((int(m), n) for m, n in audit['step_histogram'].items())
                          if any(k != nk or cut(m, q) != cut(m, nq)
                                 for (k, q), (nk, nq) in zip(pairs, mutation['pairs'])))
            write_json(EXP / f'round-{index}-proposal.json', dict(parent=pairs, mutation=mutation,
                       source=state, affected_eligible_samples=changed))
            names = [f'round-{index}-parent', f'round-{index}-child']
            futures = [pool.submit(run_phase, names[i], pp, start + 2, i,
                                   state['checkpoint'], True)
                       for i, pp in enumerate([pairs, mutation['pairs']])]
            endpoints = [f.result() for f in futures]
            parity = compare_starts(names)
            metrics = [[json.loads((EXP / 'runs' / name / 'validation' /
                        f'epoch-{epoch}' / 'metrics.json').read_text())
                        for epoch in (start + 1, start + 2)] for name in names]
            decision = selection(*metrics)
            chosen = int(decision['winner'] == 'child')
            entry = dict(round=index, parent_pairs_sixths=pairs, mutation=mutation,
                         decision=decision, start_audit=parity, metrics=metrics,
                         chosen_endpoint=endpoints[chosen])
            result_path = EXP / f'round-{index}-selection.json'
            if result_path.exists():
                assert json.loads(result_path.read_text()) == entry
            write_json(result_path, entry)
            selections.append(entry)
            state = endpoints[chosen]
            if chosen:
                pairs = mutation['pairs']
            write_json(EXP / 'search-progress.json', dict(stage='selection', epoch=state['epoch'],
                       pairs_sixths=pairs, completed_rounds=len(selections), updated=time.time()))
        state = run_phase('advance-24', pairs, 24, 0, state['checkpoint'])
        write_json(EXP / 'frozen-policy.json', dict(pairs_sixths=pairs, epoch=24,
                   checkpoint=state['checkpoint'], selections=selections))
        adaptive = run_phase('adaptive-final', pairs, 40, 0, state['checkpoint'])
        fixed = control_job.result()
    final = dict(final_pairs_sixths=pairs, selections=selections, results={})
    for name, endpoint in [('adaptive', adaptive), ('fixed-control', fixed)]:
        final['results'][name] = {}
        for split in config['evaluation_splits']:
            final['results'][name][split] = evaluate(endpoint['checkpoint'], EXP / 'final' / name / split, split)
    write_json(EXP / 'results.json', final)
    report = ['# Slow local cutpoint search', '', f'Final policy (k, numerator/6): {pairs}', '',
              '| Dataset | Fixed control | Adaptive | Difference (pp) |', '|---|---:|---:|---:|']
    for split in config['evaluation_splits']:
        a = final['results']['adaptive'][split]['accuracy']
        b = final['results']['fixed-control'][split]['accuracy']
        report.append(f'| {split} | {100*b:.2f}% | {100*a:.2f}% | {100*(a-b):+.2f} |')
    report += ['', 'Single seed11. Search used validation only. No claim of a globally optimal fixed policy.',
               'The learned policy was not retrained from scratch. See protocol.json and round-*-selection.json.']
    (EXP / 'REPORT.md').write_text('\n'.join(report) + '\n')
    write_json(EXP / 'search-progress.json', dict(stage='complete', updated=time.time(),
               final_pairs_sixths=pairs, results_file=str(EXP / 'results.json')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    EXP.mkdir(parents=True, exist_ok=True)
    lock_name = 'smoke.lock' if args.smoke else 'search.lock'
    with (EXP / lock_name).open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(EXP / ('smoke-process.json' if args.smoke else 'controller-process.json'),
                   dict(pid=os.getpid(), started=time.time()))
        try:
            smoke() if args.smoke else search()
        except BaseException as error:
            write_json(EXP / ('smoke-error.json' if args.smoke else 'search-error.json'),
                       dict(error=str(error), traceback=traceback.format_exc(), updated=time.time()))
            raise


if __name__ == '__main__':
    main()
