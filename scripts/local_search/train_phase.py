"""Run unchanged CODI training with fixed-policy data and resumable phase hooks."""
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import TrainerCallback, set_seed
from runtime import REPO, ROOT, EXP, BANK, OptionRows, main_hash, write_json, sha
from policy import validate

sys.path.insert(0, str(REPO))
import train

RUN = Path(os.environ['SEARCH_RUN'])
STOP = float(os.environ['SEARCH_STOP'])
PAIRS = validate(json.loads(os.environ['SEARCH_PAIRS']))
SMOKE = os.environ.get('SEARCH_SMOKE') == '1'
RESUME = os.environ.get('SEARCH_RESUME')
EVAL_EVERY = os.environ.get('SEARCH_EVAL_EVERY') == '1'
set_seed(11)


def state_hash(items):
    h = hashlib.sha256()
    for name, tensor in items:
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


# The unmodified Dataset loads exactly this path. Intercept only this one read;
# checkpoint, optimizer and RNG loading continue through the original torch.load.
original_load = torch.load
cache_path = Path(os.environ['CODI_DATA_CACHE'])


def load_policy_data(path, *args, **kwargs):
    if isinstance(path, (str, Path)) and Path(path) == cache_path:
        source = EXP / 'smoke-bank.pt' if SMOKE else BANK
        data = original_load(source, map_location='cpu', weights_only=False)
        suffixes = data.pop('dual_options')
        data.pop('dual_row_id')
        data['continuation_options'] = OptionRows(suffixes, PAIRS)
        return data
    return original_load(path, *args, **kwargs)


torch.load = load_policy_data


class Audit(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.metrics = {}
        self.times = []

    def on_train_begin(self, args, state, control, model=None, optimizer=None, **kw):
        assert args.seed == 11 and args.num_train_epochs == 40
        assert args.per_device_train_batch_size == 64 and args.gradient_accumulation_steps == 2
        assert args.learning_rate == .003 and args.num_latent == 6
        assert not args.fix_attn_mask and not args.include_last_cot
        assert args.continuation_weight == .3 and not model.model_args.use_decoder
        ref = json.loads((REPO / 'configs/reproduction.json').read_text())
        for name, expected in ref['core_sha256'].items():
            assert sha(REPO / name) == expected, name
        data = self.trainer.train_dataset.data_dict
        meta = dict(pairs_sixths=PAIRS, start_step=state.global_step, max_steps=state.max_steps,
                    samples=len(self.trainer.train_dataset), main_data_sha256=main_hash(data),
                    model_state_sha256=state_hash(model.state_dict().items()), resume=RESUME,
                    optimizer_has_state=bool(optimizer.state), lr=optimizer.param_groups[0]['lr'],
                    scheduler_last_epoch=self.trainer.lr_scheduler.last_epoch,
                    optimizer_state_entries=len(optimizer.state), smoke=SMOKE)
        if not SMOKE:
            assert meta['main_data_sha256'] == ref['training_reference']['main_data_sha256']
            assert meta['samples'] == 383368 and state.max_steps == 119840
        if RESUME:
            previous = json.loads((Path(RESUME) / 'trainer_state.json').read_text())
            assert state.global_step == previous['global_step'] > 0 and optimizer.state
            assert self.trainer.lr_scheduler.last_epoch == state.global_step
            meta['source_checkpoint_sha256'] = {name: sha(Path(RESUME) / name) for name in
                ['pytorch_model.bin', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json']}
        else:
            assert state.global_step == 0 and not optimizer.state
            assert meta['model_state_sha256'] == ref['training_reference']['initial_state_sha256']
        write_json(RUN / f'start-{state.global_step}.json', meta)
        self.started = time.time()
        print('SEARCH_START', json.dumps(meta), flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        for key, value in (logs or {}).items():
            if key in {'loss', 'ce_loss', 'distill_loss', 'ref_ce_loss', 'continuation_loss', 'grad_norm'}:
                if value is not None and not math.isfinite(float(value)):
                    raise FloatingPointError(f'{key}={value} at {state.global_step}')
        self.metrics.update(logs or {})
        write_json(RUN / 'progress.json', dict(step=state.global_step, epoch=state.epoch,
                   updated=time.time(), pairs_sixths=PAIRS, metrics=self.metrics))

    def on_step_begin(self, args, state, control, **kwargs):
        if len(self.times) < 20:
            torch.cuda.synchronize()
            self.step_start = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if len(self.times) < 20:
            torch.cuda.synchronize()
            self.times.append(time.time() - self.step_start)
            if len(self.times) >= 5:
                sec = sum(self.times[3:]) / len(self.times[3:])
                write_json(RUN / 'throughput.json', dict(seconds_per_step=sec,
                    estimated_40epoch_hours=sec * 119840 / 3600,
                    peak_cuda_gb=torch.cuda.max_memory_allocated() / 2**30,
                    step=state.global_step, samples=len(self.trainer.train_dataset)))

    def on_epoch_end(self, args, state, control, **kwargs):
        control.should_save = True
        if state.epoch >= STOP - 1e-6:
            control.should_training_stop = True
        return control

    def on_save(self, args, state, control, **kwargs):
        checkpoint = Path(args.output_dir) / f'checkpoint-{state.global_step}'
        epoch = round(state.epoch, 5)
        item = dict(epoch=epoch, step=state.global_step, checkpoint=str(checkpoint), pairs_sixths=PAIRS)
        write_json(RUN / 'last-checkpoint.json', item)
        if SMOKE or EVAL_EVERY or round(epoch) % 4 == 0 or epoch >= STOP - 1e-6:
            dest = RUN / 'validation' / f'epoch-{epoch:g}'
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = os.environ.get('SEARCH_EVAL_GPU', '3')
            cmd = [sys.executable, str(REPO / 'scripts/reproduce/evaluate.py'),
                   '--checkpoint', str(checkpoint), '--output', str(dest)]
            if SMOKE:
                cmd += ['--limit', '16']
            with (EXP / 'eval.lock').open('w') as lock, (RUN / f'eval-{epoch:g}.log').open('w') as log:
                fcntl.flock(lock, fcntl.LOCK_EX)
                subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            item['metrics'] = json.loads((dest / 'metrics.json').read_text())
            write_json(RUN / 'latest-evaluation.json', item)
            with (RUN / 'history.jsonl').open('a') as log:
                log.write(json.dumps(item) + '\n')
        write_json(RUN / 'saved-state.json', item)


# Keep historical sampling and loss intact; record the first two input batches
# to test identical-data resumption across the two arms.
original_compute_loss = train.CustomTrainer.compute_loss


def traced_loss(self, model, inputs, num_items_in_batch=None):
    if not hasattr(self, '_trace_batches'):
        self._trace_batches = 0
    if self._trace_batches < 2:
        record = dict(step=self.state.global_step, microbatch=self._trace_batches,
                      encoder_sha256=state_hash([('ids', inputs['encoder_input_ids'])]))
        with (RUN / 'first-batches.jsonl').open('a') as log:
            log.write(json.dumps(record) + '\n')
        self._trace_batches += 1
    return original_compute_loss(self, model, inputs, num_items_in_batch)


train.CustomTrainer.compute_loss = traced_loss
original_train = train.CustomTrainer.train


def train_with_phase(self, *args, **kwargs):
    self.add_callback(Audit(self))
    if RESUME:
        kwargs['resume_from_checkpoint'] = RESUME
    return original_train(self, *args, **kwargs)


train.CustomTrainer.train = train_with_phase
train.train()
