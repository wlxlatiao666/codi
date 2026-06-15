# Stage 2: 隐式 → 精炼隐式（自我压缩）

## 核心思想
以 Stage 1 训得的隐式模型为 Teacher，同架构、更少 latent 数的模型为 Student，进行自蒸馏压缩，进一步缩短推理时的 latent 步数。无需显式 CoT 数据。

---

## 1. 设置

| 角色 | 模型 | Latent 数 | 状态 |
|------|------|-----------|------|
| Teacher | Stage 1 的 $M_1$ | $k$ | **冻结** |
| Student | 从 $M_1$ 初始化 | $k' < k$ | 训练 |

- **不使用 Cross-Attention 模块**：Teacher 本身已是隐式，无显式 CoT 序列可对齐。
- 推荐：$k = 8, \; k' = 4$。若 Stage 1 已设 $k=4$，压缩收益有限。

---

## 2. Loss 函数

$$
\mathcal{L} = \underbrace{\mathcal{L}_{\text{CE}}^{\text{stu}}(y \mid x, z^{1:k'})}_{\text{主信号}} + \mu_1 \underbrace{\| h^{\text{ans}}_T - h^{\text{ans}}_S \|^2}_{\text{答案位对齐}} + \mu_2 \underbrace{\mathrm{KL}\!\left( P_T(y \mid \cdot) \,\|\, P_S(y \mid \cdot) \right)}_{\text{输出分布对齐}}
$$

| 项 | 权重 | 说明 |
|----|------|------|
| $\mathcal{L}_{\text{CE}}^{\text{stu}}$ | 1.0 | Student 必须能独立答对 |
| $\mu_1$（答案位对齐） | 1.0 | Teacher/Student 答案首 token 隐状态对齐 |
| $\mu_2$（KL 蒸馏） | 0.5 | 输出分布软对齐 |

---

## 3. 渐进式压缩

避免一步从 $k$ 砍到 $k'$，训练时逐步削减：

$$
k(t) = k - \left\lfloor (k - k') \cdot \frac{t}{T} \right\rfloor
$$

- Student 每步输入 $k(t)$ 个 latent，Teacher 始终用 $k$ 个。

---

## 4. 迭代扩展

可反复应用：$k \to k/2 \to k/4 \to \dots$

每轮 Student 训完冻结作为下一轮 Teacher，直到性能下降，即得最优压缩边界。
