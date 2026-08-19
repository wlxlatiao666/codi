#!/bin/bash
# CODI with fixed-segment CoT contrastive alignment (distillation-loss ablation)
# 保留 student/ref CE，只用 align loss 替代原 hidden-state distill loss
# align loss 全程固定为 0.1，同时训练 backbone 和对齐 projection heads

SAVE_DIR=/inspire/qb-ilm2/project/neosmosis/weilongxuan-253108120168/codi/checkpoints/gpt2/gsm8k-aug/lca
EXPT_NAME=gpt2-gsm8k-aug-fixed-segment-align-0.1

mkdir -p "$SAVE_DIR"

python train.py \
    --output_dir "$SAVE_DIR" \
    --expt_name "$EXPT_NAME" \
    --logging_dir "$SAVE_DIR/$EXPT_NAME/logs" \
    --logging_steps 10 \
    --model_name_or_path "/inspire/hdd/global_user/weilongxuan-253108120168/models/gpt2" \
    --data_name icot \
    --data_path "/inspire/hdd/global_user/weilongxuan-253108120168/data/GSM8k-Aug" \
    --seed 11 \
    --model_max_length 512 \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --bf16 \
    --num_train_epochs 20 \
    --learning_rate 3e-3 \
    --max_grad_norm 2.0 \
    --use_lora True \
    --lora_r 128 \
    --lora_alpha 32 \
    --lora_init \
    --save_strategy "no" \
    --save_safetensors False \
    --save_total_limit 1 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --do_train \
    --report_to "tensorboard" \
    --num_latent 6 \
    --logging_strategy "steps" \
    --use_prj True \
    --prj_dim 768 \
    --prj_dropout 0.0 \
    --distill_loss_factor 0.0 \
    --exp_mode False \
    --exp_data_num 2000 \
    --remove_eos True \
    --print_ref_model_stats True \
    --use_cross_attn_align True \
    --cross_attn_layer_idx -3 \
    --align_loss_factor 0.1 \
    --align_temperature 0.5 \
    --align_loss_warmup_ratio 0.0 \
    --align_warmup_steps 0 \
    --align_freeze_aligner_after_warmup False \
    --align_loss_decay_start 1.0
