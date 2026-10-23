# 全截点 × 全后缀续算实验（2026-10-20）

用户指定 worker：4423662（4 × H20）。远端实验目录：
`/home/tiger/codi-ab-c71db6e-20260909/all-cuts-20261020`。

## 训练目标

每个训练样本沿用历史预处理后的显式 CoT（仍移除最后一步）。若剩余 m 步，枚举非空真后缀 j=1,...,m-1；不包含整链 j=0 或仅答案 j=m。每个后缀由 EOT、剩余推理、答案、EOS 构成。有效样本的所有后缀与全部六个隐式位置 k=1,...,6 做笛卡尔积。少于两步的样本仅参与原来的主损失。

续算 CE 在一个 microbatch 内对所有 (k, sample, j, token) 有效目标求和，除以这些目标的总数。最终损失为：

`L = L_original_CODI + 0.3 * sum(CE_valid_tokens) / count(valid_tokens)`。

EOT 与 padding 不参与 CE；不同后缀共有的 token 分别计入各自预测任务。因此这是 token 等权平均，长后缀/后缀更多的样本贡献更大，非逐样本等权平均。六个位置使用相同后缀集合，归一化使每个位置总权重为 0.3/6；分支数不会放大总系数。

## 实现

- `src/all_cuts.py`：在原 CODI 每次 latent forward 完成时接入全部续算分支。缓存按来源样本复制，梯度不 detach。原主路径代码不改。
- 每次最多处理 64 条后缀，使用非重入 activation checkpoint；反向重计算保留 dropout RNG。分块控制显存与吞吐，不抽样/裁剪候选。不同分块会改变 dropout 的具体随机数分配，因此最终配置从相同初始权重重新训练，不中途切换。
- `scripts/all_cuts/train_run.py`：注入完整后缀数据和模型子类，核对初始化及原训练数据哈希，记录进度、吞吐与显存。
- `scripts/all_cuts/controller.py`：复用已验证的保存、恢复和评估流程，独立实验目录；`pairs_sixths` 兼容字段存储 all-cuts 描述字符串，不代表三点采样。
- `scripts/all_cuts/prepare.py`：核对先前逐条后缀重建审计的文件哈希，并检查包括 k=6 和 batch padding 的保守上下文上界。
- `scripts/all_cuts/launch.sh`：环境检查、单元测试、数据审计、GPU 小规模训练保存/评估，然后全量训练。

## 对齐配置

GPT-2；LoRA rank128/alpha32/dropout0.1；6 latents；projector768；seed11；bf16；batch64 × 梯度累积2；LR0.003；cosine40epoch；warmup0.03；weight decay0.1；grad clip2；383,368 条训练样本，119,840 次更新。原 attention/position 行为保持不变（fix_attn_mask=False），主数据及初始模型哈希须匹配 configs/reproduction.json。

训练使用 GPU0，评估 GPU3；每 epoch 保存完整模型/优化器/RNG，保留最近 checkpoint；每4epoch在固定512条验证集评估。最终 epoch40 自动评估 validation、GSM8K test、GSM-Hard、MultiArith、SVAMP。使用已有固定 setting 结果作历史对照，不为这次实验另起重复基线。

完整数据审计：177,390 条样本有有效后缀，共 290,203 条不同后缀；与六个隐式位置组合后，每 epoch 为 1,741,218 条辅助续算序列。最长 encoder 为 214 tokens，最长后缀为 163 tokens；最保守的 padded 上下文长度为 214+6+163=383，小于训练设置 512 和模型位置上限 1024。

## 验证与查看

服务器环境下：`python -m pytest -q tests/test_all_cuts.py tests/test_continuation.py`。覆盖枚举 loss/梯度一致、复制分支归一化、缓存隔离、checkpoint dropout/RNG 一致、六个 latent 截点覆盖、0.3 系数、无有效后缀及 Trainer 梯度累积缩放情况。

首次 GPU 试跑发现包装后的 `forward(**kwargs)` 会让 Trainer 误判 loss 已按 batch item 归一化，进而漏掉梯度累积除数。现已明确设置 `accepts_loss_kwargs=False`，并与原 CODI 的 Trainer 行为做回归测试。修正前的两步试跑归档为 `runs/smoke-before-accumulation-fix`，不用于正式训练；正式运行从原始 seed11 初始化开始。

实验目录中的 `status.json`、`runs/full/progress.json`、`runs/full/throughput.json`、`runs/full/train.log` 是实时状态；`data-audit.json` 为数据审计；最终结果写入 `results.json`。每 epoch 保存后可从最后完整 checkpoint 恢复；运行同一 launch.sh 会复用已有 checkpoint。

该实验检验广覆盖辅助监督的效果，并不学习截点映射。不同截点同时预测多个后缀，可能产生目标竞争；效果由最终独立评估判定。

## 启动实测

2026-10-20，最终 chunk_size=64 的正式运行已通过前20次更新：约1.732秒/update，峰值已分配显存26.015 GiB；119,840次更新线性估计57.67小时，不含保存、评估及后续批次波动。16条分块的性能试跑归档在 `runs/full-chunk16-benchmark`；最终运行未从性能试跑恢复。全量数据哈希和初始模型哈希均通过基线一致性检查。

本地 `experiments/all-cuts-20261020/` 保存启动审计与吞吐快照，它不是实时状态或最终效果结果。最终运行以远端 `runs/full` 为准。
