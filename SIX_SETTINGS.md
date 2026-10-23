# 六种训练 setting 与结果归档（2026-10-23）

这些入口复用已经运行过的训练实现；本次整理没有修改训练损失、数据预处理或公共超参。
当前 GPU1 / GPU2 的任务继续运行，不要在它们尚未退出时重复执行对应启动脚本。

## 环境与启动

在 Linux H20 worker 上运行，Python 3.11，依赖见 `requirements.txt`。

```bash
export CODI_ASSET_ROOT=/home/tiger/codi-ab-c71db6e-20260909
cd "$CODI_ASSET_ROOT/repository"
```

| setting | 训练入口 | 默认训练 GPU | 方法 |
|---|---|---:|---|
| Base CODI | `GPU=0 bash scripts/train_codi.sh` | 0 | 原始答案、teacher 与 latent 蒸馏损失 |
| CODI + SIM-CoT | `GPU=1 bash scripts/train_sim.sh` | 1 | 独立解释 decoder，解释损失权重 1 |
| 固定截点续算 | `GPU=2 bash scripts/train_continuation.sh` | 2 | `(3,1/3),(4,1/3),(5,2/3)`，每个 optimizer step 选一组；权重 0.3 |
| 所有 latent × 所有后缀 | `bash scripts/train_all_cuts.sh` | 0 | k=1…6 × 所有 proper suffix；全体有效 token 的 CE 均值 ×0.3 |
| 顺序均分、预测单步 | `bash scripts/train_aligned_step.sh` | 1 | j=ceil(km/6)，仅该步及 EOS，不含答案；各 k 的 token CE 均值各乘 0.05 |
| 最多三个连续后缀起点 | `bash scripts/train_window_suffix.sh` | 2 | 中心 ceil(km/6)，边界整体平移，最多三个 j；预测步骤 j…m 及答案；按 k、样本、候选、token 分层平均 ×0.3 |

前三项可通过 `GPU` / `CODI_RUN_ROOT` 指定设备和输出根目录。后三项通过 `ALL_CUTS_ROOT`、`ALIGNED_STEP_ROOT`、`WINDOW_SUFFIX_ROOT` 指定实验根目录，设备配置保留历史值。
**入口有状态恢复逻辑：使用已有目录会检查、恢复或跳过已有阶段，不意味着从头训练。** 新实验应使用独立空目录；在空闲设备上逐个运行，勿同时启动占用同一卡的入口。

当前单步实验原控制器指定 GPU2 评测，通过 `scripts/reproduce/eval_routes.json` 将该运行目录的评测路由到 GPU3，并与其他任务串行。
这个路由是针对当前绝对路径的：换目录或机器后，需先按实际目录配置路由及共享锁；不要将路由锁设置成控制器已持有的同一个锁（会死锁）。参见 `tests/test_eval_routing.py`。

## 控制变量

实际启动记录已由 `python scripts/reproduce/audit_settings.py` 审计，结果见 `results/settings-audit.json`。

| 公共参数 | 值 |
|---|---|
| 模型 / seed | GPT-2 / 11 |
| 主训练数据 | 383,368 条；主数据哈希相同 |
| epoch / optimizer updates | 40 / 119,840 |
| microbatch / accumulation / 有效 batch | 64 / 2 / 128 |
| learning rate / scheduler / warmup | 0.003 / cosine / 0.03 |
| weight decay / clip norm | 0.1 / 2 |
| LoRA rank / alpha / dropout | 128 / 32 / 0.1 |
| latent 数 / projector | 6 / 768维，dropout=0 |
| 精度 / 最大序列长度 | bf16 / 512 |
| 蒸馏 / ref loss 权重 | 1 / 1 |
| attention / teacher 尾步 | fix_attn_mask=False / include_last_cot=False |

三个新续算方法的初始模型哈希与基准一致。历史启动的 checkpoint 保留数（1 或 2）和日志详略不同，不改变优化目标。
SIM-CoT 独立 decoder、解释损失，以及不同续算方法的目标、有效样本、归一化和随机数消耗属于方法差异；不能宣称六种方法的逐步随机轨迹完全一致。
固定/全后缀方法需要至少两步（177,390 条有效辅助样本）；单步/窗口方法接受至少一步（320,527 条）。主训练数据和总更新数不因此改变。

固定截点有两个历史运行：`fixed_original` 是原版从头训练；`fixed_control` 是局部搜索实验的固定策略对照，从共同 epoch8 warmup 恢复至40。
它们采用相同截点与公共超参，但不是同一个 checkpoint。`train_continuation.sh` 对应原版入口；历史最优 fixed-control 的结果与来源单独保留，不将两者混用。

## 必需外部资产（未上传 Git）

为了不改变既有训练的数据口径，以下服务器目录仍保留：

- `assets/models/gpt2`、`assets/datasets/ood-20260916`、`venv`
- `continuation-search-20260930/runs/{base-codi,random-cut}/dataset.pt`
- `continuation-dual-random-20261010/dataset.pt`
- `continuation-search-20260930/data/validation`、`gsm8k-aug-id-20261014/data`、`multiarith-full-20261012/data`
- `random-disturb-20261015/{data-audit.json,smoke-bank.pt}`
- 单步/窗口数据准备还引用 `aligned-step-20261021/data-audit.json`

仅 clone GitHub 不会自动得到这些精确缓存、数据和权重；需保留或复制这些资产。`preflight.py` 校验核心源码、缓存哈希及依赖版本。

## 准确率和热力图

- [准确率归档说明](results/README.md)：包含所有保留的历史 `metrics.json`，以及运行中实验截至快照时的验证结果。
- `results/provenance/`：启动参数、数据哈希、状态快照；不是实时进度。
- [Base CODI 完整 k-r](experiments/base-codi-cut-probe-20261020/train_stratified-heatmap.png)
- [固定截点完整 k-r](experiments/fixed-codi-cut-probe-20261021/train_stratified-heatmap.png)
- [Base / 固定截点完整 k-j](experiments/cut-probe-absolute-j/train_stratified-all-lengths.png)
- [新模型完整 k-r / k-j](experiments/all-cuts-cut-probe-20261023/train_stratified-all-lengths-kr-kj.png)

每个探针目录同时保留验证集图、固定链长图、PNG/PDF、CSV 数值表、manifest 和压缩逐目标 loss。未测过的模型不伪造热力图。
探针中的 j 是跳过的显式步骤数（预测 j+1…m），与单步/窗口训练中“首个预测步骤”的一基 j 定义不同。
绘图依赖单列于 `requirements-probe.txt`；训练依赖保持原版本。

```bash
python scripts/cut_probe/run.py --model all-cuts --output /path/to/new-probe
python scripts/cut_probe/analyze.py /path/to/new-probe
# plot_single.py 读取 per_target.csv.gz；先 gzip -k 原始 CSV
python scripts/cut_probe/plot_single.py /path/to/new-probe \
  --reference experiments/base-codi-cut-probe-20261020
```

## 整理范围

移除了本地过期日志副本、旧诊断逐样本中间文件、旧 Trainer 状态副本及 Python 缓存；远端仅清理仓库内可再生缓存和 AppleDouble 元数据。
没有删除或移动在训目录、权重、优化器/RNG 状态、数据缓存或评测锁。旧 SIM-CoT 权重在此前清理中已删除，本仓库保留其历史准确率记录，不包含可恢复的权重。
