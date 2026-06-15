# Stage 1: 显式 → 隐式（Cross-Attention 软对齐蒸馏）

## 核心思想
训练时引入轻量 Cross-Attention 模块，让 Student 的每个 latent token 软对齐 Teacher 显式 CoT 的对应片段，提供更密集的监督信号；推理时丢弃该模块，Student 仅靠主干网络生成答案，零额外开销。

---

## 1. 输入定义

| 角色 | 输入序列 | 说明 |
|------|----------|------|
| Teacher | $[x] \; c_1 \dots c_N \; [ans] \; y$ | 显式 CoT，$N$ 个推理 token |
| Student | $[x] \; \langle\text{bot}\rangle \; z_1 \dots z_k \; \langle\text{eot}\rangle \; [ans] \; y$ | $k$ 个连续 latent thought |

- $x$：问题；$y$：答案
- $z_i$：Student 第 $i$ 个 latent 位置的**输入表示**（continuous embedding）
- $h^T_t$：Teacher 在 CoT 第 $t$ 个位置的最后一层隐状态
- $h^S_{z_i}$：Student 在第 $i$ 个 latent 位置的**中间层隐状态**（**取倒数第 3–4 层**，非最后一层）
- $h^{\text{ans}}$：生成答案首 token 时的隐状态

> **注意**：$z_i$ 与 $h^S_{z_i}$ 不同。前者是输入层的 latent 表示，后者是 transformer 中间层输出；Cross-Attention 的 Query 来自 $z_i$，软对齐 Loss 对齐的是 $h^S_{z_i}$。

---

## 2. Cross-Attention 模块

**仅在训练时使用，推理时丢弃。**

### 结构
一层 cross-attention，无 FFN，无残差：

$$
\alpha_{i,t} = \mathrm{softmax}_t\!\left( \frac{(W_Q z_i)^\top (W_K h^T_t)}{\sqrt{d_r}} \right), \quad
\tilde{h}^T_i = \sum_{t=1}^{N} \alpha_{i,t} \cdot (W_V h^T_t)
$$

- Query：来自 Student latent $z_i$
- Key/Value：来自 Teacher CoT 隐状态 $h^T_t$

### 低秩约束（必须）

$W_Q = U_Q V_Q, \; U_Q \in \mathbb{R}^{d \times r}, \; V_Q \in \mathbb{R}^{r \times d_r}$，$W_K, W_V$ 同理。

| 超参 | 值 |
|------|-----|
| 模型隐维度 $d$ | 4096 (7B) |
| 低秩维度 $r$ | 64 |
| 注意力头数 | 4 |
| Head 维度 $d_r$ | 64 |
| 层数 | **1** |
| FFN / 残差 | **无** |

**目的**：模块容量足够弱，无法独立完成对齐，迫使监督压力回流到 $z_i$。

---

## 3. Loss 函数

$$
\mathcal{L} = \underbrace{\mathcal{L}_{\text{CE}}^{\text{tea}}(y \mid x, c)}_{\text{Teacher 路径}} + \underbrace{\mathcal{L}_{\text{CE}}^{\text{stu}}(y \mid x, z)}_{\text{主信号}} + \lambda_g \underbrace{\| h^{\text{ans}}_T - h^{\text{ans}}_S \|^2}_{\text{全局对齐}} + \lambda_a \underbrace{\frac{1}{k}\sum_{i=1}^{k}\| \tilde{h}^T_i - h^S_{z_i} \|^2}_{\text{软对齐}}
$$

| 项 | 权重 | 说明 |
|----|------|------|
| $\mathcal{L}_{\text{CE}}^{\text{stu}}$ | 1.0 | 主信号，Student 必须能独立答对 |
| $\mathcal{L}_{\text{CE}}^{\text{tea}}$ | 1.0 | Teacher 路径同步训练 |
| $\lambda_g$（全局对齐） | 1.0 | CODI 原项，答案位隐状态对齐 |
| $\lambda_a$（软对齐） | **0.2 – 0.3** | 辅助信号，必须小于 $\lambda_g$ |

---

## 4. 训练调度

$\lambda_a$ 采用 warm-up + decay：

```
λ_a(t) = 0,            t ∈ [0, t_warm]
       = ramp to 0.3,  t ∈ [t_warm, t_peak]
       = decay to 0,   t ∈ [0.8T, T]
```

**关键点**：训练末期 $\lambda_a \to 0$，让模型适应推理时无模块的状态。

---

## 5. 关键工程细节

1. **Teacher 隐状态 detach**：$h^T_t$ 不参与反向传播，保持 Teacher 路径稳定。
2. **中间层对齐**：$h^S_{z_i}$ 取倒数第 3–4 层，避免对齐 loss 直接污染输出 logits。
3. **推理时丢弃模块**：Student 推理路径只走主干 transformer。

---

## 6. 可行性验证（必须）

| 设置 | 推理时 cross-attn | 目的 |
|------|-------------------|------|
| A | **丢弃**（主方案） | 报告最终性能 |
| B | 保留（输出注入主干） | 验证模块未吸收关键信息 |

**要求**：A 与 B 性能差 $\Delta \approx 0$，证明模块仅为梯度引导，可安全丢弃。
