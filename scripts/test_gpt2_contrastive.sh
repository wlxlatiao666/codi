#!/bin/bash
# CODI align-only auxiliary-loss ablation test script
# 对应 train_gpt2_gsm8k-aug-contrastive.sh 的 20-epoch checkpoint

CKPT_DIR="/inspire/qb-ilm2/project/neosmosis/weilongxuan-253108120168/codi/checkpoints/gpt2/gsm8k-aug/lca/gpt2-gsm8k-aug-align-only-0.1/gpt2/ep_20/lr_0.003/seed_11"

python test.py \
    --data_path "/inspire/hdd/global_user/weilongxuan-253108120168/data/gsm8k" \
    --data_name "gsm8k" \
    --output_dir "$CKPT_DIR" \
    --model_name_or_path "/inspire/hdd/global_user/weilongxuan-253108120168/models/gpt2" \
    --seed 11 \
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
    --inf_latent_iterations 6 \
    --inf_num_iterations 1 \
    --remove_eos True \
    --use_lora True \
    --ckpt_dir "$CKPT_DIR" \
    --use_cross_attn_align True \
    --cross_attn_layer_idx -3
