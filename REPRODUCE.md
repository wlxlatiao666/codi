# 复现当前三组正式实验

根目录 `train.py`、`src/model.py`、`src/continuation.py`、`src/search_data.py` 原样迁自 `attention-aligned-20261012/source`。四个文件的 SHA-256 固定在 `configs/reproduction.json`，训练启动前强制校验。SIM-CoT 是同一模型的 `use_decoder` 开关，续算为 `continuation_weight` 和缓存中的三个固定截点选项。

## 环境与数据

在 4 卡 H20 worker 内运行。沿用 `/home/tiger/codi-ab-c71db6e-20260909/venv/bin/python`；核心环境 Python3.11、torch2.6.0+cu124、transformers4.52.4、peft0.15.2、accelerate1.7.0。`requirements.txt` 已更新为本轮实测的主要依赖版本；worker 的完整环境快照见 `configs/environment-freeze.txt`。优先复用既有 venv；新安装环境仍需通过 preflight 和回归验证。

远端整理后的代码仓位于 `/home/tiger/codi-ab-c71db6e-20260909/repository`，数据与模型仍在其上级目录。本地同一代码仓为 `/Users/bytedance/codes/codi`。本地不保存训练数据或模型。

以下资源必须保留：

| 资源 | 相对 `CODI_ASSET_ROOT` 路径 |
|---|---|
| GPT-2 基座和 tokenizer | `assets/models/gpt2` |
| CoDi / SIM-CoT 训练缓存 | `continuation-search-20260930/runs/base-codi/dataset.pt` |
| 固定截点续算训练缓存 | `continuation-search-20260930/runs/random-cut/dataset.pt` |
| 自定义512条验证集 | `continuation-search-20260930/data/validation` |
| 公开 GSM8K-Aug test | `gsm8k-aug-id-20261014/data` |
| GSM-Hard / SVAMP | `assets/datasets/ood-20260916/{gsm-hard,svamp}` |
| MultiArith600 | `multiarith-full-20261012/data` |

训练入口固定复用历史预处理缓存，不静默重新下载或重新切分数据。缓存文件哈希、383,368条输入的 tensor 哈希和共享初始化权重哈希均有校验；缺失或变化会报错。续算缓存包含完整固定截点目标与有效样本筛选结果，不能用普通 GSM8K-Aug 文件替代。

## 启动三组训练

在 worker 的代码仓目录运行，三个终端分别执行：

```bash
bash scripts/train_codi.sh          # GPU0
bash scripts/train_sim.sh           # GPU1
bash scripts/train_continuation.sh  # GPU2
```

GPU3 用于验证和最终 ID 评测，评测任务通过文件锁串行。脚本前台运行，可按需放入 tmux。三组配置固定为 seed11、40epochs、batch64×累积2、LR0.003 等；完整协议见 `configs/reproduction.json`。启动脚本只是路径和生命周期封装，实际 Trainer/模型实现保持原样。

默认新结果存入 `/home/tiger/codi-ab-c71db6e-20260909/reorganized-runs-20261014/runs/{codi,sim,continuation}`，不会写入历史正式结果目录。每epoch保存，保留最近2份checkpoint，每4epoch测验证集，最终测ID。已存在 `last-checkpoint.json` 时自动恢复完整优化器/RNG状态；已完成40epoch则跳过。重新从零训练需换一个空的 `CODI_RUN_ROOT`。

可用环境变量调整文件位置与卡号（不改变算法超参）：

```bash
export CODI_ASSET_ROOT=/home/tiger/codi-ab-c71db6e-20260909
export CODI_RUN_ROOT=$CODI_ASSET_ROOT/my-new-reproduction
export CODI_PYTHON=$CODI_ASSET_ROOT/venv/bin/python
export CODI_EVAL_GPU=3
GPU=0 bash scripts/train_codi.sh
```

不要把 `CODI_RUN_ROOT` 指向历史实验目录。单卡运行时训练进程仍占显存，建议保留独立评测卡。

## 评测已有最终模型

```bash
GPU=0 bash scripts/reproduce/evaluate_all.sh codi \
  /home/tiger/codi-ab-c71db6e-20260909/attention-aligned-20261012/runs/codi/run/gpt2/ep_40/lr_0.003/seed_11/checkpoint-119840 \
  /home/tiger/codi-ab-c71db6e-20260909/my-evaluation/codi
```

SIM-CoT 将方法与 checkpoint 路径中的 `codi` 换为 `sim`；续算换为 `continuation`。脚本依次测 ID、GSM-Hard、MultiArith600、SVAMP1000，保持历史 batch128、greedy、6 latent、256答案token和MultiArith分批边界。SIM-CoT解释decoder仅用于训练，评测加载时移除其权重。

## 防止整理引入偏差

- 四个训练核心文件与历史正式代码逐字节一致。
- `scripts/reproduce/verify_parity.py` 对照历史模型：训练态启用dropout，在验证进程内启用确定性算子，检查初始化、前向、全部梯度、RNG状态与一次AdamW更新；训练入口仍保留原有确定性设置。覆盖CoDi、SIM-CoT和三个续算截点。
- 使用三个历史最终checkpoint复测四个数据集，对比逐题预测。
- 整理不意味着重新完成三组40epoch训练；完整训练的数值重复性还依赖相同GPU、依赖、输入顺序和运行环境。
- `analysis/`（以及误拼的 `analyst/`）均被Git忽略；必要配置与结果文档独立保留。旧实验脚本和 `EXPERIMENT_B.md` 仅作历史参考，不适用于当前根目录接口。旧模型接口测试已归档，当前辅助函数测试及GPU回归脚本替代其验证职责。

验证记录见 `configs/reorganization-verification.json`：5项CPU测试通过，三种方法（续算覆盖3个截点）的确定性前向/梯度/一次更新逐位相同，12组评测逐题预测相同。最初在默认GPU执行模式下，续算反向和更新没有逐位一致；验证进程启用确定性算子后差异消失。此开关只用于代码对照，未加入正式训练，因此不承诺重新训练40epoch得到逐位相同的权重。

为保留四个核心文件的原始哈希，本次没有清理这些历史文件已有的尾随空格和不可达兼容代码。
