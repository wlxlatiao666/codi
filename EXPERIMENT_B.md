# Experiment B: single-path continuation supervision

This implements an auxiliary teacher-forced suffix loss using the existing
`zen-E/GSM8k-Aug` (`--data_name icot`) or `zen-E/GSM8k-Aug-NL`
(`--data_name icot-full`) data. No new dataset, generated branches, external
teacher, or API is needed. Pretrained model/data downloads still use the
official training pipeline.

## Objective

For an existing formatted CoT with m steps, split at
`j = clamp(ceil(m * continuation_cut_ratio), 1, m-1)`. At latent checkpoint
`k = clamp(ceil(num_latent * continuation_cut_ratio), 1, num_latent-1)`, fork the
student KV cache and decode:

```
main:      Q -> BOT -> z_1 ... z_k ... z_n -> EOT -> answer
auxiliary:                 cache fork    -> EOT -> r_(j+1) ... r_m -> answer
```

The auxiliary branch never receives r_1...r_j as text. Its loss is next-token
cross entropy over the remaining CoT and answer (including terminal EOS),
normalized by the total number of supervised tokens in the batch. EOT, an
optional separator EOS, and padding have labels -100. The first suffix token
IS supervised from the last marker. The auxiliary cache is cloned without
detaching gradients; correct continuation text never enters the main cache.

`L = original_CODI_loss + continuation_loss_factor * suffix_CE`

This is a fixed-ratio checkpoint heuristic, not an explicit proof that z_k
matches teacher step j. It adds no intermediate hidden-state alignment, output
KL, free-running rollout loss, or counterfactual branching. The existing
teacher CE and answer-position distillation remain in place. Inference needs
no auxiliary decoder or continuation branch.

## Data and coverage

Auxiliary steps are derived from the SAME formatted CoT used by the baseline,
after its existing `include_last_cot` processing. By default the original final
CoT step remains excluded; the auxiliary target still ends with the answer.
Recognized boundaries: standalone `<<equation>>` sequences, nonempty lines,
sentence-ending punctuation followed by whitespace, and space-separated
equations whose every element contains `=`. This is conservative formatting
logic, not a semantic parser; inspect data examples when adding a new format.

Examples with fewer than two retained steps are kept for the original losses
but get no auxiliary loss. The same applies to targets longer than
`continuation_max_tokens`, combined auxiliary sequences beyond the context
limit, or baseline question/CoT/answer segments exceeding the official 256-token
tokenization cap. Auxiliary targets are never silently truncated. Coverage and
skip reasons are logged once at dataset construction; no eligible examples is
an error. An entirely ineligible minibatch has finite zero auxiliary loss.

## Reproducible comparison

```
# A: matched baseline (new branch disabled)
CONTINUATION_LOSS_FACTOR=0 bash scripts/train_experiment_b.sh

# B: same settings, auxiliary loss enabled
CONTINUATION_LOSS_FACTOR=0.3 bash scripts/train_experiment_b.sh

# Short integration run on a CUDA training host
NUM_EPOCHS=1 BATCH_SIZE=2 bash scripts/train_experiment_b.sh \
  --exp_mode True --exp_data_num 32 --max_steps 2 --report_to none
```

The script defaults to GPT-2 and bf16 on a compatible CUDA host; set MODEL_NAME,
DATA_NAME, OUTPUT_DIR, SEED, BATCH_SIZE, NUM_EPOCHS or append training arguments.
Changing model family requires choosing suitable learning rate/projection size,
not merely changing MODEL_NAME. Default batch 8 with accumulation 16 gives the
same single-device effective batch 128 as the original GPT-2 example. The new
branch adds training time and activation memory; there is no claim of equal
training FLOPs. Report wall time/memory and compare at matched budgets too.

New flags:

| Flag | Default | Meaning |
|---|---|---|
| continuation_loss_factor | 0.0 | Zero disables data construction and branch execution |
| continuation_cut_ratio | 0.5 | Split fraction for both retained CoT and latent checkpoint |
| continuation_max_tokens | 256 | Auxiliary decoder sequence length limit, including markers |

Log `continuation_loss` is unweighted; total training loss includes its weighted
value. The original test.py inference path is unchanged. Evaluation must use
the same model/projection/latent settings as training.

## Baseline corrections and limits

Experiment B requires `--fix_attn_mask True`. This flag now propagates masks
and explicit position IDs through encoder, latent iterations, and answer
decoder, including left-padded examples. BOTH A and B scripts enable it. The
legacy default `fix_attn_mask=False`, `continuation_loss_factor=0` path is
preserved. Comparisons with old checkpoints/settings must account for the mask
correction. The tokenizer's official 256-token cap is not globally changed.

Trainer.compute_loss now supports the Transformers return_outputs/
num_items_in_batch calling convention; construction uses processing_class
instead of the removed tokenizer argument on newer Transformers. Non-LoRA construction initializes
lora_config to None instead of leaving it undefined. Existing teacher double
forward and model-name-dependent answer-position handling are retained.

## Validation

```
python -m pytest -q tests/test_continuation.py
```

Offline CPU tests construct random tiny GPT-2 and Llama models: suffix-only
targets, marker/padding masking, ineligible rows, total loss composition, cache
isolation, direct auxiliary gradient flow into prefix states, left-padding
invariance, actual dataset/collator integration, and two optimizer steps through
CustomTrainer. They do not
establish GSM8k accuracy or GPU/distributed training performance. Run the short
integration command on the intended CUDA environment before a full experiment.
