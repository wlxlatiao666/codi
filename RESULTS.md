# 三种 latent 推理方法的实验结果

本表对应 2026-10-12 启动、2026-10-14 完成的 attention/位置编码对齐实验。模型均为 GPT-2，单 seed11，最终 epoch40 / step119840 checkpoint；未用测试集挑选 checkpoint。

| 方法 | GSM8K-Aug test（1,319） | GSM-Hard（1,319） | MultiArith（600） | SVAMP（1,000） |
|---|---:|---:|---:|---:|
| CoDi | 41.02%（541） | 9.33%（123） | 93.00%（558） | 41.60%（416） |
| CoDi + SIM-CoT | 41.85%（552） | 9.93%（131） | 92.67%（556） | 42.00%（420） |
| 固定截点组合续算 | 42.08%（555） | 9.17%（121） | 93.33%（560） | 42.30%（423） |

括号内为答对题数。续算在 ID、MultiArith、SVAMP 上数值最高；SIM-CoT 在 GSM-Hard 上最高。单 seed 的小幅差异不能作为稳定优势或统计显著性的结论。

## 数据与评测口径

- 训练使用 GSM8K-Aug，经本实验模板去重、验证集隔离及长度筛选后实际输入 383,368 条；验证集为自行划分的 512 条。不是对论文私有清洗数据逐条复现。
- ID 采用 SIM-CoT 官方 `Coconut/preprocessing/gsm_icot.bash` 指向的公开 iCoT/GSM8K-Aug `test.txt`。已验证 1,319 个题目、顺序和数值答案与此前原始 GSM8K test 一致，独立重测的逐题预测也完全一致。作者 CoDi 脚本中的私有 `gsm_test_clean.json` 不公开，未验证该文件一致性。
- GSM-Hard：`juyoung-trl/gsm-hard`，revision `fb5ed446258f12c5e46cb27de4a00fd4934ee1b5`，全部 1,319 条。
- MultiArith：`ChilleD/MultiArith`，revision `144d44c3fb87c0b9097ac9593c789e716a282e3e`，train420 + test180，不去重；分别在两个 split 内以 batch128 评测，保留 420 的分批边界。
- SVAMP：`ChilleD/SVAMP`，revision `5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef`，train700 + test300。
- 所有模型使用相同数据、顺序和解码：6 个 latent，greedy，batch128，最多 256 个答案 token，比较提取的最后一个数值；本轮没有截断样本。位置编码保留官方物理位置语义，因此批次组成属于评测协议，不应随意改变。

## 方法设置

统一 LoRA r128/alpha32/dropout0.1、batch64×累积2、LR0.003、cosine、warmup0.03、weight decay0.1、max grad norm2、bf16、40 epochs。

SIM-CoT 使用官方独立解释解码器，辅助权重1，保留官方 step padding 行为。续算权重0.3，组合为 `(latent=3, CoT比例=1/3)`、`(4,1/3)`、`(5,2/3)`；每个优化器步用 `random.Random(11 + 1000003 * step)` 等概率取一个，同一累积步内共用组合。教师及续算目标都删除最后一个 CoT 步骤，续算有效样本177,390（46.27%），首步权重1，缓存 clone 不 detach，不隔离辅助分支的 RNG。

## 结果来源

远端根目录：`/home/tiger/codi-ab-c71db6e-20260909`。

- 权重：`attention-aligned-20261012/runs/{codi,sim,continuation}/run/gpt2/ep_40/lr_0.003/seed_11/checkpoint-119840`。
- ID重测：`gsm8k-aug-id-20261014/results/{方法}/gsm8k-aug/metrics.json`。
- OOD：`ood-aligned-20261014/results/{方法}/{数据集}/metrics.json`。
- 公开ID数据：[固定版本 test.txt](https://raw.githubusercontent.com/da03/Internalize_CoT_Step_by_Step/e06a32ee5e4cd117171daeb4755d2a97ece62761/data/gsm8k/test.txt)。

本次代码整理保留上述历史结果；整理后的回归验证另存，不混作重新训练的结果。
