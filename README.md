# CoDi / SIM-CoT / 固定截点续算

当前根目录实现对应 2026-10-12 attention/位置编码对齐后的三组正式实验。

- [复现实验与启动方式](REPRODUCE.md)
- [GSM8K 与 OOD 实验结果](RESULTS.md)
- 共用模型：`src/model.py`；续算损失：`src/continuation.py`；数据/截点选项：`src/search_data.py`。
- `--use_decoder True` 启用 SIM-CoT；固定截点续算通过 `--continuation_weight 0.3` 和冻结的数据缓存启用。
- 请使用 `scripts/train_codi.sh`、`scripts/train_sim.sh`、`scripts/train_continuation.sh`。训练依赖远端保留的精确数据缓存。
- `analysis/` 是不纳入 Git 的历史实验归档。下方原始说明及旧脚本仅作历史参考，不是本轮复现入口。

---

# CODI

This is the official implementation of the paper: [CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation](https://arxiv.org/abs/2502.21074).
Accepted by EMNLP 2025 🎉


![codi](imgs/codi_method_v4.png)

## Setup

**Clone the Repository**:
```
git clone git@github.com:zhenyi4/CODI.git
cd CODI
```

**Environment Setup**:
```
conda create --name codi python=3.12
conda activate codi
pip install -r requirements.txt
```

## Running the Results

Pretrained model weights are available at https://huggingface.co/zen-E, including:
* zen-E/CODI-gpt2
* zen-E/CODI-llama3.2-1b-Instruct
 
To evaluate accuracy on GSM8K, run:
```
bash script/test_gpt2.sh # or script/test_llama1b.sh
```
You can change the data_name argument to "svamp", "gsm-hard", or "multi-arith" to evaluate on out-of-distribution (OOD) mathematical benchmarks. 

## Interpret Latent Thoughts (for Section 5 in the Paper)

To probe and visualize latent thoughts on GSM8K, run:
```
bash scripts/probe_latent_token.sh
```
The output file will be saved in outputs/.

## Training
Optional single-path continuation supervision (Experiment B), using the same
GSM8k data, is documented in [EXPERIMENT_B.md](EXPERIMENT_B.md). Run
`bash scripts/train_experiment_b.sh`; set `CONTINUATION_LOSS_FACTOR=0` for the
matched baseline. The feature is disabled by default in `train.py`.

**GSM8k-Aug**
```
bash scripts/train_gpt2_gsm8k-aug.sh # or train_llama1b_gsm8k-aug.sh
```

**GSM8k-Aug-NL**
```
bash scripts/train_gpt2_gsm8k-aug-nl.sh
```

**Commonsense**
```
bash train_gpt2_commonsense.sh # or train_llama_commonsense.sh
```
You can change the testing script's data_name to "commonsense" for evaluation.

## Key Arguments
* `use_prj`: Whether use a projection layer for the last layer hidden state.

* `prj_dim`: The dimension of the hidden state of the projection layer.

* `prj_no_ln`: Whether the projection layer is not followed by a LayerNorm layer.

* `distill_loss_div_std`: Whether divide the distillation loss via the standard deviation of the teacher's hidden states.

* `distill_loss_type`: The type of loss used for distillation (e.g. l1, l2, smoothl1).

* `distill_loss_factor`: The multiplier that scales the distillation loss in the total loss calculation.

* `ref_loss_factor`: The multiplier that scales the teacher's cross entropy loss in the total loss calculation.

* `num_latent`: The number of latent thoughts used for training.

* `inf_latent_iterations`: The number of latent thoughts used for inference.

* `include_last_cot`: Include the last CoT step in the training data.

* `fix_attn_mask`: Flag to fix a known attention mask bug (leave as False by default).

* `max_token_num`: Discards training samples that exceed this token length threshold.

## Citation
If you use this code base in your research, please cite our paper with the following BibTex entry:
```bibtex
@article{shen2025codi,
      title={CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation}, 
      author={Zhenyi Shen and Hanqi Yan and Linhai Zhang and Zhanghao Hu and Yali Du and Yulan He},
      year={2025},
      journal={arXiv preprint arxiv:2502.21074},
}
```
