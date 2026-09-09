#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Use identical settings with CONTINUATION_LOSS_FACTOR=0 for the matched A run.
CONTINUATION_LOSS_FACTOR="${CONTINUATION_LOSS_FACTOR:-0.3}"
python train.py \
  --output_dir "${OUTPUT_DIR:-./outputs/continuation_ab}" \
  --expt_name "single_path_w${CONTINUATION_LOSS_FACTOR}" \
  --model_name_or_path "${MODEL_NAME:-gpt2}" \
  --data_name "${DATA_NAME:-icot}" \
  --seed "${SEED:-11}" \
  --model_max_length 512 \
  --per_device_train_batch_size "${BATCH_SIZE:-8}" \
  --gradient_accumulation_steps 16 \
  --bf16 True \
  --num_train_epochs "${NUM_EPOCHS:-40}" \
  --learning_rate 3e-3 \
  --max_grad_norm 2.0 \
  --use_lora True --lora_init True --lora_r 128 --lora_alpha 32 \
  --save_strategy epoch --save_total_limit 2 --save_safetensors False \
  --weight_decay 0.1 --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --report_to tensorboard --logging_steps 10 \
  --num_latent 6 --use_prj True --prj_dim 768 \
  --distill_loss_type smooth_l1 --distill_loss_div_std True \
  --distill_loss_factor 1 --ref_loss_factor 1 \
  --remove_eos True --fix_attn_mask True \
  --continuation_loss_factor "$CONTINUATION_LOSS_FACTOR" \
  --continuation_cut_ratio 0.5 --continuation_max_tokens 256 \
  --print_loss False --print_ref_model_stats False \
  "$@"
