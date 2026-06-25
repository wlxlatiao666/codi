#!/bin/bash
# Evaluate the same CODI checkpoint on SVAMP, gsm-hard, and MultiArith sequentially.

CKPT_DIR="/inspire/qb-ilm2/project/neosmosis/weilongxuan-253108120168/codi/checkpoints/gpt2/gsm8k-aug/crossattn/gpt2-gsm8k-aug-crossattn/gpt2/ep_40/lr_0.003/seed_11"
MODEL_PATH="/inspire/hdd/global_user/weilongxuan-253108120168/models/gpt2"

# Common arguments shared across all datasets
COMMON_ARGS=(
    --model_name_or_path "$MODEL_PATH"
    --seed 11
    --model_max_length 512
    --bf16
    --lora_r 128
    --lora_alpha 32
    --lora_init
    --batch_size 128
    --greedy True
    --num_latent 6
    --use_prj True
    --prj_dim 768
    --prj_no_ln False
    --prj_dropout 0.0
    --inf_latent_iterations 6
    --inf_num_iterations 1
    --remove_eos True
    --use_lora True
    --ckpt_dir "$CKPT_DIR"
)

# 1. SVAMP
echo "========================================"
echo "Evaluating on SVAMP..."
echo "========================================"
python test.py \
    --data_path "/inspire/hdd/global_user/weilongxuan-253108120168/data/SVAMP" \
    --data_name "svamp" \
    --output_dir "$CKPT_DIR" \
    "${COMMON_ARGS[@]}"

# 2. gsm-hard
echo ""
echo "========================================"
echo "Evaluating on gsm-hard..."
echo "========================================"
python test.py \
    --data_path "/inspire/hdd/global_user/weilongxuan-253108120168/data/gsm-hard" \
    --data_name "gsm-hard" \
    --output_dir "$CKPT_DIR" \
    "${COMMON_ARGS[@]}"

# 3. MultiArith
echo ""
echo "========================================"
echo "Evaluating on MultiArith..."
echo "========================================"
python test.py \
    --data_path "/inspire/hdd/global_user/weilongxuan-253108120168/data/MultiArith" \
    --data_name "multi-arith" \
    --output_dir "$CKPT_DIR" \
    "${COMMON_ARGS[@]}"

echo ""
echo "All OOD evaluations finished."
