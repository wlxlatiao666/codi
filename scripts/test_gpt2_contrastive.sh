#!/bin/bash
# CODI with Learnable Contrastive Alignment Test Script
# 使用可学习的对比对齐的 CODI 测试脚本

CKPT_DIR="/inspire/qb-ilm2/project/neosmosis/weilongxuan-2531080168/codi/checkpoints/gpt2/gsm8k-aug/lca/gpt2-gsm8k-aug-contrastive/gpt2/ep_40/lr_0.003/seed_11"

python test.py \
    --data_name "gsm8k" \
    --output_dir "$CKPT_DIR" \
    --model_name_or_path "/inspire/hdd/global_user/weilongxuan-2531080168/models/gpt2" \
    --seed 42 \
    --model_max_length 512 \
    --bf16 \
    --lora_r 128 \
    --lora_alpha 32 \
    --lora_init \
    --batch_size 16 \
    --greedy True \
    --num_latent 6 \
    --use_prj True \
    --prj_dim 768 \
    --prj_no_ln False \
    --prj_dropout 0.0 \
    --inf_latent_iterations 1 \
    --inf_num_iterations 1 \
    --remove_eos True \
    --use_lora True \
    --ckpt_dir "$CKPT_DIR" \
    --use_cross_attn_align True \
    --cross_attn_layer_idx -1
