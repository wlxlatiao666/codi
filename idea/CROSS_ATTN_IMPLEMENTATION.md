# Cross-Attention 软对齐蒸馏实现说明

## 概述

本文档说明在 CODI 基础上实现的 Cross-Attention 软对齐蒸馏方案。

## 核心思想

训练时引入轻量 Cross-Attention 模块，让 Student 的每个 latent token 软对齐 Teacher 显式 CoT 的对应片段，提供更密集的监督信号；推理时丢弃该模块，Student 仅靠主干网络生成答案，零额外开销。

## 文件修改清单

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `src/model.py` | 大幅修改 | 新增 `CrossAttentionAligner` 类、`get_align_loss_weight` 函数，修改 `CODI` 类的 `__init__` 和 `forward` 方法 |
| `train.py` | 小幅修改 | 修改数据预处理和 DataCollator 以支持 CoT 位置信息传递 |
| `scripts/train_gpt2_cross_attn.sh` | 新增 | 带 Cross-Attention 的训练脚本示例 |
| `scripts/test_gpt2_cross_attn.sh` | 新增 | 测试脚本示例 |

## 新增组件详解

### 1. `CrossAttentionAligner` 类

位置：`src/model.py`

**作用**：轻量级 cross-attention 模块，用于计算 Student latent 与 Teacher CoT 的软对齐。

**架构**：
```
Student z_i (query)        Teacher CoT h^T_t (key/value)
      |                              |
      v                              v
Low-rank projection (UQ*VQ)    Low-rank projection (UK*VK, UV*VV)
      |                              |
      +--------------+---------------+
                     |
                     v
            Multi-head attention
                     |
                     v
            Low-rank projection (UO*VO)
                     |
                     v
            Aligned Teacher representation \tilde{h}^T_i
```

**关键设计**：
- 所有投影矩阵都使用低秩分解 (W = U * V)
- 无 FFN，无残差连接
- 注意力权值计算后对 Teacher 侧做加权平均

**低秩约束**：
- 低秩维度 `cross_attn_rank` 默认为 64
- 注意力头数 `cross_attn_heads` 默认为 4
- 每个头的维度为 `rank // heads`

### 2. `get_align_loss_weight` 函数

位置：`src/model.py`

**作用**：计算对齐损失权重的调度函数。

**调度策略**：
```
λ_a(t) = 0,                              t ∈ [0, t_warm]
         ramp to max_weight,             t ∈ [t_warm, t_peak]
         max_weight,                     t ∈ [t_peak, t_decay_start]
         decay to 0,                     t ∈ [t_decay_start, T]
```

**目的**：
- 早期：不给对齐损失，让模型先学好基本任务
- 中期：加入对齐损失，提供密集监督
- 后期：逐渐减小到 0，让模型适应推理时无 cross-attention 的状态

### 3. `TrainingArguments` 新增参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_cross_attn_align` | False | 是否启用 cross-attention 软对齐 |
| `cross_attn_rank` | 64 | 低秩投影维度 |
| `cross_attn_heads` | 4 | 注意力头数 |
| `cross_attn_layer_idx` | -3 | Student 对齐的层索引 (负数表示从后数) |
| `align_loss_factor` | 0.3 | 对齐损失最大权重 λ_a |
| `align_loss_warmup_steps` | 500 | Warmup 步数 |
| `align_loss_peak_steps` | 2000 | 达到峰值的步数 |
| `align_loss_decay_start` | 0.8 | 开始 decay 的训练进度比例 |

### 4. `CODI.forward` 新增逻辑

**数据流向**：
```
1. Student 编码问题得到初始隐状态 z_0
   ↓
2. 迭代 latent 步骤：
   - 保存每个步骤的输入 embedding (用于 cross-attention query)
   - 保存指定层的隐状态 (用于对齐损失计算)
   ↓
3. Teacher 前向传播得到 CoT 区间隐状态
   ↓
4. Cross-Attention 对齐：
   - Query: Student latent embeddings
   - Key/Value: Teacher CoT hidden states
   - 得到对齐后的 Teacher 表示 \tilde{h}^T_i
   ↓
5. 计算软对齐损失 L_align = MSE(h^S_{z_i}, \tilde{h}^T_i)
   ↓
6. 总损失 L_total = L_CE + L_distill + L_ref_CE + λ_a * L_align
```

## 训练-推理不对称设计

### 训练时
```python
if self.use_cross_attn_align:
    # 1. 收集 Student latent 状态
    student_latent_inputs.append(latent_embd)
    student_latent_hiddens.append(layer_hidden)

    # 2. Cross-Attention 对齐
    aligned_repr, attn_weights = self.cross_attn_aligner(
        query=student_queries,
        key_value=teacher_cot_hiddens.detach()
    )

    # 3. 计算对齐损失
    align_loss = self.align_loss_fct(student_hiddens, aligned_repr.detach())
```

### 推理时
**完全不需要修改**！`test.py` 直接复用原代码，`cross_attn_aligner` 模块存在但不会被调用。

## 关键工程细节

### 1. Teacher 隐状态 detach
```python
aligned_repr, attn_weights = self.cross_attn_aligner(
    query=student_queries,
    key_value=teacher_cot_hiddens.detach()  # 重要！
)
```
确保 Teacher 侧不会通过对齐损失更新。

### 2. 中间层对齐
```python
self.cross_attn_layer_idx = -3  # 倒数第 3 层
layer_hidden = outputs.hidden_states[self.cross_attn_layer_idx]
```
- 不对齐最后一层，避免直接污染输出 logits
- 中间层有更丰富的中间表示

### 3. CoT 区间提取
```python
# 从 ref_input_ids 中提取 CoT 区间
cot_start = len(source_ids)      # 问题结束，CoT 开始
cot_end = len(source_ids) + len(cot_ids)  # CoT 结束，答案开始
```

### 4. 可变长度 CoT 的 padding
```python
# Padding 到 batch 内最大长度
for h, m in zip(teacher_cot_hiddens, teacher_cot_mask):
    pad_len = max_cot_len - h.shape[0]
    padded_h = torch.cat([h, torch.zeros(pad_len, h.shape[1])], dim=0)
    padded_m = torch.cat([m, torch.zeros(pad_len, dtype=torch.bool)], dim=0)
```

## 使用方法

### 训练

1. **启用 Cross-Attention**：
```bash
python train.py \
    --use_cross_attn_align True \
    --align_loss_factor 0.3 \
    --cross_attn_rank 64 \
    --cross_attn_heads 4 \
    --cross_attn_layer_idx -3 \
    ... (其他参数)
```

2. **使用提供的脚本**：
```bash
chmod +x scripts/train_gpt2_cross_attn.sh
bash scripts/train_gpt2_cross_attn.sh
```

### 推理

**无需任何修改**，直接使用原测试脚本或：
```bash
chmod +x scripts/test_gpt2_cross_attn.sh
bash scripts/test_gpt2_cross_attn.sh
```

### 超参数调优建议

| 场景 | 推荐设置 |
|------|----------|
| 小模型 (GPT-2) | `cross_attn_rank=64`, `cross_attn_heads=4` |
| 大模型 (LLaMA-7B) | `cross_attn_rank=128`, `cross_attn_heads=8` |
| 容易过拟合 | 减小 `align_loss_factor` (如 0.1-0.2) |
| 欠拟合 | 增大 `align_loss_factor` (如 0.3-0.5) |
| 长训练任务 | 增大 `align_loss_decay_start` (如 0.9) |

## 损失组成

训练日志会包含以下损失项：
```
loss: 总损失
ce_loss: Student 答案生成 CE 损失
distill_loss: 答案位置隐状态蒸馏损失
ref_ce_loss: Teacher CoT+答案生成 CE 损失
align_loss: Cross-Attention 软对齐损失 (新增)
align_weight: 当前对齐损失权重 (新增)
```

## 验证方案有效性

可以按照 `stage1.md` 的建议进行验证：

| 实验 | 推理时 cross-attn | 预期性能 |
|------|-------------------|----------|
| A (主实验) | 丢弃 (默认) | 报告最终性能 |
| B (验证) | 保留并注入 | 应该与 A 相近 (Δ ≈ 0) |

如果 B 显著优于 A，说明 cross-attention 模块吸收了关键信息，可能需要：
- 减小模块容量 (降低 rank)
- 增大 align_loss_decay_start
- 提前把 align_loss 降到 0

## 向后兼容性

代码完全向后兼容：
- 不设置 `use_cross_attn_align` 时，行为与原始 CODI 完全一致
- 旧的检查点可以直接加载
- 所有新增参数都有合理的默认值

## 代码增量统计

| 指标 | 数值 |
|------|------|
| 新增代码行数 | ~350 行 |
| 修改代码行数 | ~150 行 |
| 主要新增类 | 1 个 (`CrossAttentionAligner`) |
| 主要新增函数 | 1 个 (`get_align_loss_weight`) |
| 新增参数 | 9 个 |
