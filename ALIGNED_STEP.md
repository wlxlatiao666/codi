# 顺序均分的单步显式监督实验

Worker 4423662（4×H20）：GPU1训练、GPU2评估。GPU0上的全后缀实验与其GPU3评估保持原样。新目录 `/home/tiger/codi-ab-c71db6e-20260909/aligned-step-20261021`。

## 映射与目标

令 K=6，m为原预处理后保留的显式CoT步骤数（沿用baseline移除最后一步的规则）。对k=1..6，预测步骤编号j=ceil(k*m/6)，j从1开始。此处j是目标步骤编号，不是之前热力图中的已跳过步数。例如m=6映射[1,2,3,4,5,6]；m=2映射[1,1,1,2,2,2]；m=4映射[1,2,2,3,4,4]。m>6时仅有六个对应步骤，不额外增加latent；m=1六个位置预测同一步；m=0只训练原主任务。

在前k个latent完成后复制其KV cache，不detach梯度，输入 `EOT + 第j步文本 + EOS`，teacher forcing只监督该步及终止EOS，EOT和padding不计loss。不提供真实显式前缀，不包含后续步骤、换行后的后续链或`The answer is:`答案。主任务仍正常训练最终答案。

原EOT入口保持不变，不新增token、decoder或提示。因此k=6的EOT后会同时有主任务回答和辅助单步预测的训练压力，这是本实验的已知任务歧义；不假定已实现自由生成时的任务切换。

## 权重

每个k先独立计算microbatch中有效目标token的平均CE，记作L_k。总目标为：

`L_total = L_original_CODI + (0.3/6) * (L_1+...+L_6)`。

六个位置分别归一化，各位置系数严格0.05。不把所有k的token混成一个总平均，以免单步长度差异改变各位置权重。一个位置内token等权，非样本等权。

## 对照配置与实施

与之前实验同初始化、主数据、seed11、GPT2+LoRA rank128/alpha32、6latent、projector768、bf16、batch64×accumulation2、LR0.003、cosine40epoch、warmup0.03、weight decay0.1、clip2。383368条主训练样本，119840次更新。从原始初始化开始，不从base或固定续算的已训练模型继续。原attention/position规则保持fix_attn_mask=False。

`src/aligned_step.py`在原主路径每个latent forward后接入单步分支；复用已验证的cache复制与非重入activation checkpoint，chunk64。`accepts_loss_kwargs=False`保留Trainer梯度累积的原缩放。每epoch保存完整状态、仅保留最近checkpoint；每4epoch评估512条验证数据；epoch40后评估validation、GSM8K test、GSM-Hard、MultiArith和SVAMP。

数据从历史主输入中重建全部保留步骤，逐步encode/decode检查，禁止答案模板出现在辅助目标，验证主数据哈希不变，并审计原m>=2样本数及链长分布。与旧的真后缀目标相比，m=1样本现在也可参与辅助监督，覆盖率差异会在data-audit.json记录。

## 验证与启动

单元测试覆盖m=1/2/4/6/8映射、空样本、不同目标长度下六个位置等权、完整forward与逐分支独立计算的loss/梯度一致、主路径logits不变、梯度累积缩放，以及复用实现的checkpoint RNG与cache隔离测试。随后运行256样本GPU smoke训练、保存、评估，成功才启动正式训练。

启动：`bash scripts/aligned_step/launch.sh`。

状态：`launch.log`、`status.json`、`data-audit.json`、`runs/full/progress.json`、`runs/full/throughput.json`、`runs/full/last-checkpoint.json`。最终结果为`results.json`。本文件描述实验方案，不代表训练已完成。

## 全量数据审计结果

383368条主训练样本哈希与baseline一致。320527条样本参与辅助监督，其中143137条仅有一个保留步骤；62841条无保留步骤。原先m>=2的177390条样本和链长分布均核对一致。每epoch共有1923162条单步辅助序列（6×有效样本），最长目标65 tokens，最保守padded上下文285 tokens。数据审计包含m=1/2/6的实际步骤与映射例子，见本地`experiments/aligned-step-20261021/data-audit.json`。

15项CPU单元测试通过，256样本GPU smoke训练、保存和评估已完成，正式训练进程于2026-10-21启动。实际进度与吞吐以远端运行记录为准。

正式训练启动核验：前40次更新已运行，loss及六个位置的单步CE均为有限值；原始模型初始化哈希与baseline一致。前20次更新测得1.538秒/update，40epoch线性预估51.21小时（不含保存/评估及后续批次波动），峰值CUDA已分配26.02GiB。本地`experiments/aligned-step-20261021/`保存该启动快照，非实时状态。
