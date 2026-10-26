# 单截点 (3, 1/2) 对齐版重训

本轮从与当前 CODI / all-cuts 相同的初始化开始训练，不加载早期单截点权重。

- 入口：`bash scripts/train_single_cut.sh`，默认 GPU0。
- 根目录：`$CODI_ASSET_ROOT/single-cut-k3-r50-20261023`，可用 `SINGLE_CUT_ROOT` 覆盖。
- k=3；j=min(m-1,max(1,ceil(m/2)))；目标为 EOT + 显式步骤 j+1…m + answer + EOS。
- 仅一个候选；续算 token CE 权重0.3；不对缓存 detach；主数据383,368条，公共有效辅助样本口径沿用固定截点数据。
- seed11、40epoch、119,840 updates、batch64 × accumulation2、LR0.003 cosine、warmup0.03、weight decay0.1、clip2、LoRA r128/alpha32、6 latent、bf16、fix_attn_mask=False。
- 从 `continuation-dual-random-20261010/dataset.pt` 选择精确后缀，不重新生成或截断目标。
- 复用 `scripts/local_search/train_phase.py` 的数据、审计和断点逻辑；只在新训练子进程中将三截点策略验证器替换为仅接受 `[[3,3]]`（比例以六分之一为单位）的验证器。
- smoke 使用独立输出目录；正式训练的模型/优化器重新初始化。启动审计强制比对主数据与初始模型哈希。
- 每epoch保存模型、优化器、scheduler及RNG，保留最近一个完整checkpoint；每4epoch验证，最终评测validation/test/GSM-Hard/MultiArith全集/SVAMP。
- GPU3评测通过 `all-cuts-20261020/eval.lock` 串行。只给新实验目录建立链接，不改变已有实验的锁。
- worker回收后，在新worker挂载相同目录，确认GPU0空闲后重新运行入口，可从最后完整epoch恢复。不会自动分配新worker。

当前其他实验代码没有修改。运行状态、训练进度和吞吐量分别写入实验目录的 `status.json`、`runs/full/progress.json`、`runs/full/throughput.json`。
