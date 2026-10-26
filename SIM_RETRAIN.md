# CODI + SIM-CoT 重训（2026-10-23）

原 SIM-CoT checkpoint 已被清理。在现有服务器 checkpoint、本地中转目录及 Hugging Face 账户备份中未找到完整副本，因此新建独立实验，不从其他方法的训练权重续训。

## 启动与输出

在 worker 4423662 的仓库目录执行 `bash scripts/train_sim_retrain.sh`。默认输出目录为 `/home/tiger/codi-ab-c71db6e-20260909/sim-retrain-20261023`，训练使用物理 GPU3。重复执行前必须确认旧控制器已退出；控制器也持有排他训练锁。

先使用独立的 256 样本缓存完成训练、保存及验证检查，随后启动完整训练。正式训练复用未修改的 `scripts/reproduce/train_entry.py` 和原始模型实现。

| 参数 | 值 |
| --- | --- |
| 基座 | GPT-2 |
| 方法 | CODI + SIM-CoT，独立 decoder |
| 主训练样本数 | 383,368 |
| epoch / 总更新步数 | 40 / 119,840 |
| seed | 11 |
| batch / 梯度累积 | 64 / 2 |
| 学习率 | 0.003，cosine，warmup 0.03 |
| LoRA | r=128，alpha=32，dropout=0.1 |
| latent 数量 | 6 |
| 解释损失权重 | 1 |
| 续算损失权重 | 0 |
| 精度 | bf16 |

每个 epoch 保存 checkpoint，保留最近两份；每四个 epoch 验证，训练完成后评测 validation、GSM8K test、GSM-Hard、MultiArith full 和 SVAMP。checkpoint 包括优化器与随机数状态，重新启动时从最近完整 checkpoint 恢复。

GPU3 同时承担其他实验的评测，因此新训练进程的 PyTorch 显存分配上限为 80%，评测继续共用 `all-cuts-20261020/eval.lock` 串行执行。显存限制不改变训练超参；计算资源共享可能使评测或训练变慢。

状态记录为 `sim-controller-status.json`；正式训练审计、进度和速度分别为 `runs/sim/fresh-start.json`、`runs/sim/progress.json`、`runs/sim/throughput.json`。完整训练耗时应以该次速度测量为准。worker 生命周期可能不足以完成 40 epoch，届时需要在新 worker 上从共享云盘 checkpoint 继续。
