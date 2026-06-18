#!/bin/bash
# CODI with Cross-Attention Soft Alignment Test Script
# 使用 Cross-Attention 软对齐的 CODI 测试脚本

CKPT_DIR="./outputs/codi_gpt2_cross_attn"

python test.py \
    --data_name "gsm8k" \
    --output_dir "$CKPT_DIR" \
    --model_name_or_path "gpt2" \
    --seed 42 \
    --model_max_length 512 \
    --bf16 \
    --lora_r 128 \
    --lora_alpha 32 \
    --lora_init \
    --batch_size 128 \
    --greedy True \
    --num_latent 6 \
    --use_prj True \
    --prj_dim 768 \
    --prj_no_ln False \
    --prj_dropout 0.0 \
    --inf_latent_iterations 6 \
    --inf_num_iterations 1 \
    --remove_eos True \
    --use_lora True \
    --ckpt_dir "$CKPT_DIR" \
    \
    # 注意：推理时不需要 cross_attn 参数，因为模块会被自动忽略
