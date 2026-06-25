#!/bin/bash
# CODI with Cross-Attention Soft Alignment Training Script
# 使用 Cross-Attention 软对齐的 CODI 训练脚本

SAVE_DIR=/inspire/qb-ilm2/project/neosmosis/weilongxuan-253108120168/codi/checkpoints/gpt2/gsm8k-aug/crossattn

mkdir -p "$SAVE_DIR"

# 复制当前脚本到输出目录以便复现
# cp "$0" "$SAVE_DIR"

python train.py \
    --output_dir "$SAVE_DIR" \
    --expt_name gpt2-gsm8k-aug-crossattn \
    --logging_dir "$SAVE_DIR/logs" \
    --logging_steps 10 \
    --model_name_or_path "/inspire/hdd/global_user/weilongxuan-253108120168/models/gpt2" \
    --data_path "/inspire/hdd/global_user/weilongxuan-253108120168/data/GSM8k-Aug" \
    --seed 11 \
    --model_max_length 512 \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --bf16 \
    --num_train_epochs 40 \
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
    --distill_loss_div_std True \
    --exp_mode False \
    --exp_data_num 2000 \
    --remove_eos True \
    --print_ref_model_stats True \
    \
    # ===== Cross-Attention 软对齐参数 =====
    --use_cross_attn_align True \
    --cross_attn_rank 64 \
    --cross_attn_heads 4 \
    --cross_attn_layer_idx -3 \
    --align_loss_factor 0.3 \
    --align_loss_warmup_steps 500 \
    --align_loss_peak_steps 2000 \
    --align_loss_decay_start 0.8
