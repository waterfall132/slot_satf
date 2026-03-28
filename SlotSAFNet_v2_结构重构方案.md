# SlotSAFNet v2 结构重构方案

## 1. 文档目标

本文档在 `SlotSAFNet_改进思路.md` 的基础上，进一步给出一个 **可落地的 SlotSAFNet v2 重构方案**，目标是：

1. 让当前代码更贴近论文方法；
2. 解决当前版本中的关键瓶颈；
3. 为后续代码修改、实验验证和论文撰写提供统一路线图。

---

## 2. v2 设计原则

SlotSAFNet v2 的核心原则如下：

### 原则 1：Query 表征必须稳定
query 的前景表示不能过度依赖 support episode 的 prototype，否则会引入 episode-specific bias。

### 原则 2：Slot 模块必须可学习
slot 不能只是 soft clustering center，而必须具备一定的可学习能力，能够在训练中逐步学到更合理的前景/背景分解。

### 原则 3：Foreground / Background 都要有结构化表示
不仅要有前景 prototype，也要有显式背景 prototype，否则 background 分支很难支撑论文中的核心 claim。

### 原则 4：分类主任务优先，辅助损失服务于分类
主任务仍然是 few-shot classification，所有额外模块和损失都必须围绕提升分类判别力，而不是独立堆叠复杂性。

---

## 3. 当前版本到 v2 的核心变化

| 模块 | 当前版本 | v2 改进方向 |
|---|---|---|
| Query masking | coarse prototype average conditioned | 完全 class-agnostic / self-seeded |
| Slot update | 线性投影 + weighted mean | residual + MLP update |
| Slot init | prototype + noise | support: prototype-seeded；query: self-seeded / learnable global seed |
| Prototype type | fg only | fg + bg |
| Logits | fg + weak bg term | fg/bg 结构化联合判别 |
| Contrastive loss | fg triplet + bg repulsion | fg prototype triplet + bg prototype separation |
| Regularization | 基本无 | slot diversity + mask sparsity |

---

## 4. v2 模型总流程

## 4.1 Support 分支
对于每个类的 support 图像：

1. backbone 提取局部特征；
2. 每张 support 图单独执行 Slot-SAF；
3. 得到每张图的 soft mask；
4. 计算 foreground pooled feature 与 background pooled feature；
5. 聚合所有 support 图，得到：
   - `p_fg_n`
   - `p_bg_n`

最终每个 episode 中有：

- foreground prototypes: `P_fg = {p_fg_1, ..., p_fg_N}`
- background prototypes: `P_bg = {p_bg_1, ..., p_bg_N}`

---

## 4.2 Query 分支
每张 query 图像仅做 **一次** Slot-SAF 分解，不依赖候选类原型：

1. backbone 提特征；
2. query 自身计算 seed（如 descriptor 均值）；
3. slot refinement；
4. 得到统一：
   - `q_fg`
   - `q_bg`

### 注意
query 分支不应该为每个类单独生成表示。
这样可以保证：
- 推理阶段一致；
- query 表征稳定；
- 避免 support-conditioned bias。

---

## 4.3 分类决策
对每个 query 与所有类 prototype 比较：

### 前景匹配分支
- query foreground vs foreground prototypes

### 背景抑制分支
可以有两种候选方式：

#### 方式 A（推荐起步）
使用 query foreground 与 background prototypes 的距离：

- 离前景原型越近越好
- 离背景原型越远越好

即：

- `d_fg(n) = d(q_fg, p_fg_n)`
- `d_bg(n) = d(q_fg, p_bg_n)`

logit 可以设计为：

```text
logit_n = -gamma_fg * d_fg(n) + gamma_bg * d_bg(n)
```

#### 方式 B
利用 query background 与背景 prototype 的一致性：

```text
logit_n = -gamma_fg * d(q_fg, p_fg_n) - gamma_bg * d(q_bg, p_bg_n)
```

但这种方式可能更难调，建议先从方式 A 开始。

---

## 5. Slot 模块重构建议

## 5.1 v2 Slot 模块的目标
不是完全照搬原始 Slot Attention，而是构建一个 **适合 few-shot descriptor decomposition 的轻量可学习 Slot 模块**。

---

## 5.2 建议结构

### 输入
局部特征：
- `X ∈ R^{B × L × C}`

### 初始化
#### Support
- 使用 coarse prototype seed + noise

#### Query
- 使用 self-seeded mean feature
- 或 learnable global slot seed

---

### 迭代更新
每轮做：

1. `Q = W_q(X)`
2. `K = W_k(S)`
3. `V = W_v(X)`
4. `A = softmax(QK^T / sqrt(C))`
5. `U = A^T V`
6. `S = S + MLP(LN(U))`

### 可选增强
- residual connection
- slot normalization
- update gating
- LayerNorm

---

## 5.3 为什么要加 residual/MLP
当前直接：

```python
slots = updates / attn_sum
```

太像 soft k-means center update，学习能力有限。

加上：
- residual update
- MLP refinement

可以让 slot：
- 更稳定；
- 具备一定非线性建模能力；
- 学到更复杂的局部模式分解。

---

## 5.4 slot diversity 正则
建议加一个 diversity loss，避免多个 slot 收敛成一样。

### 可选形式
#### attention-based
约束不同 slot attention map 低相关：

```text
L_div = ||A^T A - I||_F^2
```

#### feature-based
约束 slot 表征之间 cosine similarity 低：

```text
L_div = mean_{u!=v} cos(s_u, s_v)
```

推荐先用 feature-based，更容易实现。

---

## 6. foreground / background prototype 设计

## 6.1 support 端 prototype 构造
对于类 n：

### foreground prototype
```text
p_fg_n = weighted sum of support foreground descriptors
```

### background prototype
```text
p_bg_n = weighted sum of support background descriptors
```

都应使用 soft mask 做归一化加权平均，而不是简单 mean。

---

## 6.2 background prototype 的意义
它不是为了代替 foreground 分类，而是提供：

1. 额外判别信息；
2. 更强的辅助对比约束；
3. 更贴近论文中的 fg/bg decoupling 叙述。

---

## 7. v2 损失函数设计

## 7.1 分类损失
必须保留：

```text
L_cls = CrossEntropy(logits, y)
```

这是主损失。

---

## 7.2 foreground triplet loss
仍然保留 hardest-negative triplet：

```text
L_fg_triplet = [m + d(q_fg, p_fg_y) - d(q_fg, p_fg_y_minus)]_+
```

其中 hardest negative 从当前 episode 中选择。

---

## 7.3 background separation loss
推荐在 v2 中正式引入。

### 方案 A（推荐）
query foreground 应远离正类背景原型：

```text
L_bg_sep = [m + d(q_fg, p_fg_y) - d(q_fg, p_bg_y)]_+
```

### 方案 B
query background 应接近正类背景原型：

```text
L_bg_align = d(q_bg, p_bg_y)
```

### 方案 C
同时用 A + B。

起步建议：
- 先加 `L_bg_sep`
- 如果稳定，再加 `L_bg_align`

---

## 7.4 mask regularization
为了防止 mask 全图泛化或极端塌缩，建议加入：

### mask sparsity
```text
L_mask = mean(M)
```

### mask smoothness（可选）
如果保留 spatial 结构可视化，可以加 TV regularization。

---

## 7.5 diversity regularization
```text
L_div = slot diversity loss
```

---

## 7.6 推荐总损失
推荐从如下形式起步：

```text
L_total = L_cls + λ1 * L_fg_triplet + λ2 * L_bg_sep + λ3 * L_div + λ4 * L_mask
```

### 初始权重建议
- `λ1 = 0.1 ~ 0.2`
- `λ2 = 0.05 ~ 0.1`
- `λ3 = 0.01 ~ 0.05`
- `λ4 = 0.001 ~ 0.01`

建议先保守，不要让辅助项压过 CE。

---

## 8. 训练策略建议

## 8.1 两阶段训练

### Stage 1
冻结 backbone，只训练：
- slot module
- prototype head / classifier head

持续 5~10 epoch。

### Stage 2
解冻 encoder 最后一层 block，小学习率联合训练。

---

## 8.2 学习率建议
- backbone: `1e-5 ~ 5e-5`
- slot/head: `1e-4 ~ 5e-4`

如果 backbone 已经很好，不建议完全自由微调全网络。

---

## 8.3 episode 数量建议
当前 10000 episodes/epoch 容易过拟合。

建议试：
- 2000
- 4000
- 6000

并观察：
- val 是否更稳定
- 早停点是否更清晰

---

## 8.4 1-shot 和 5-shot 分开调
Slot-SAF 在 1-shot 和 5-shot 场景可能行为不同：

- 1-shot：prototype 噪声大，slot module 更容易被 support 偏差放大
- 5-shot：prototype 更稳，foreground purification 可能更有效

建议两种设置分别调超参数，不要共用一套。

---

## 9. v2 消融实验计划

## 9.1 基础对比
必须先跑：

1. ProtoNet baseline
2. 当前 SlotSAFNet v1
3. SlotSAFNet v2

---

## 9.2 结构消融
建议做：

### A. Query seed 方式
- support prototype average
- query self-seeded
- learnable global slot seed

### B. 是否有 background prototype
- fg only
- fg + bg

### C. slot update 方式
- weighted mean only
- residual + MLP update

### D. loss 组合
- `L_cls`
- `L_cls + L_fg_triplet`
- `L_cls + L_fg_triplet + L_bg_sep`
- 完整版

### E. slot 数量与迭代轮数
- `V ∈ {2,4,6,8}`
- `T ∈ {1,2,3,5}`

---

## 10. 论文支撑实验建议

## 10.1 soft mask 可视化
至少展示：
- support image + mask
- query image + mask
- fg / bg response

---

## 10.2 slot map 可视化
最好展示不同 slot 是否真的学到不同语义部分。

如果多个 slot 图几乎一样，说明 diversity 不够。

---

## 10.3 错误案例分析
对比：
- ProtoNet 正确 / SlotSAFNet 错误
- ProtoNet 错误 / SlotSAFNet 正确

看 slot 模块到底帮了什么、又在哪些场景失效。

---

## 11. 代码重构建议

建议后续把模型代码拆成以下结构：

```text
models/
  slot_saf_net_v2.py
  modules/
    slot_attention.py
    mask_generator.py
    prototype_builder.py
    losses.py
```

### 好处
- slot 模块、mask 模块、prototype 构造、loss 解耦；
- 更方便做 ablation；
- 更便于论文和代码对应。

---

## 12. 推荐实施顺序

### Step 1（最优先）
改 query 端：
- self-seeded / global slots

### Step 2
构造 `bg_protos`

### Step 3
引入更强 slot update

### Step 4
加 `L_bg_sep`

### Step 5
加 diversity / mask regularization

### Step 6
做消融与可视化

---

## 13. 一句话总结

SlotSAFNet v2 的核心不是简单“继续调参”，而是完成三件事：

1. **让 query 表示真正稳定**；
2. **让 slot 模块真正具备学习能力**；
3. **让 background 分支从“辅助概念”变成“结构化模型组成部分”**。

如果这三点做到位，Slot-SAFNet 才有可能真正超过 ProtoNet baseline，而不是继续停留在“理论上更复杂、结果上差不多”的状态。
