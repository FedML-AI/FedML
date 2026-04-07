# VeriFL v16 → v18f 升级方案

> **文档定位**：本文档是《重生推进方案》的附录文档，详细说明将 FedML 中的 VeriFL 实现从当前 v16 时代算法升级到 v18f（Relative Delta-Norm）变体的完整差异分析和实施计划。
>
> **源算法规格**：`/algrothm.md`（VeriFL-v18f 完整算法说明文档）
>
> **结论**：v16→v18f 涉及三阶段防御流水线中 **Phase 1（适应度函数）** 和 **Phase 2（投影→裁剪）** 的根本性算法变更，以及 **Phase 3（动量）** 的 BN 感知修复。这不是简单的参数调整，而是算法级升级。

---

## 目录

- [1. 变更全景](#1-变更全景)
- [2. Phase 1 差异：适应度函数](#2-phase-1-差异适应度函数)
- [3. Phase 2 差异：投影 → 裁剪](#3-phase-2-差异投影--裁剪)
- [4. Phase 3 差异：动量 BN 感知](#4-phase-3-差异动量-bn-感知)
- [5. Phase 4 差异：BN 重校准](#5-phase-4-差异bn-重校准)
- [6. GPU 加速器变更](#6-gpu-加速器变更)
- [7. 超参数映射](#7-超参数映射)
- [8. 代码级变更清单](#8-代码级变更清单)
- [9. 实施优先级与依赖关系](#9-实施优先级与依赖关系)
- [10. 与重生推进方案的关系](#10-与重生推进方案的关系)
- [11. v16→v18f 演化脉络](#11-v16v18f-演化脉络)

---

## 1. 变更全景

### 1.1 三阶段变更摘要

| 阶段 | v16（FedML 当前） | v18f（目标） | 变更级别 |
|------|-------------------|-------------|---------|
| **Phase 1: GA 搜索** | 绝对模型范数正则 `λ=0.1` | 相对增量范数正则 `λ_f=32.0` | 🔴 **算法变更** |
| **Phase 2: 投影/裁剪** | 全模型范数双边投影（scale 可 >1 或 <1） | 增量空间单边裁剪（仅压缩，永不放大） | 🔴 **算法变更** |
| **Phase 3: 动量** | 全参数动量（含 BN buffers） | BN 感知动量（BN buffers 跳过） | 🟠 **Bug 修复**（已在重生推进方案 D-3 中） |
| **Phase 4: BN 重校准** | 在验证集上刷新 BN running stats | 移除 | 🟠 **移除**（已在重生推进方案 D-2 中） |

### 1.2 影响评估

- **Phase 1 变更影响**：GA 搜索的评分标准完全改变。v16 惩罚的是"模型参数有多大"，v18f 惩罚的是"相对于上轮改了多少"。语义从"参数控制"转为"变化控制"。
- **Phase 2 变更影响**：从"把所有人统一到参考半径"变为"只压缩超标者"。v16 会放大弱更新客户端（可能放大噪声），v18f 保留弱更新原始信号。
- **Phase 3 + 4 变更**：已在重生推进方案中规划，此处不重复论证。

---

## 2. Phase 1 差异：适应度函数

### 2.1 代价函数对比

**v16（当前实现）**：

$$
\mathcal{C}_{v16}(\boldsymbol{\alpha}) = L_{val}(\mathbf{W}(\boldsymbol{\alpha})) + \lambda \cdot \|\mathbf{W}(\boldsymbol{\alpha})\|_2
$$

其中 $\lambda = 0.1$（`lambda_reg`），$\|\mathbf{W}(\boldsymbol{\alpha})\|_2$ 是聚合模型的**绝对模型范数**（通过 `model.parameters()` 计算，包含所有可训练参数）。

**v18f（目标）**：

$$
\mathcal{C}_{v18f}(\boldsymbol{\alpha}) = L_{val}(\mathbf{W}(\boldsymbol{\alpha})) + \lambda_f \cdot \frac{\|\boldsymbol{\Delta}(\boldsymbol{\alpha})\|_{\mathcal{T}}}{\|\mathbf{W}_{t-1}\|_{\mathcal{T}} + \varepsilon_r}
$$

其中：
- $\lambda_f = 32.0$
- $\boldsymbol{\Delta}(\boldsymbol{\alpha}) = \mathbf{W}(\boldsymbol{\alpha}) - \mathbf{W}_{t-1}$（增量）
- $\|\cdot\|_{\mathcal{T}}$ 仅在可训练参数上计算
- $\varepsilon_r = 10^{-8}$

### 2.2 关键差异分析

| 维度 | v16 | v18f |
|------|-----|------|
| **正则项语义** | 惩罚"参数绝对大小" | 惩罚"相对于上轮的变化量" |
| **参考基准** | 无（绝对范数） | 上一轮全局模型 $\mathbf{W}_{t-1}$ |
| **量纲适应性** | 固定 $\lambda=0.1$，对模型尺度敏感 | $\lambda_{eff} = \lambda_f / \|\mathbf{W}_{t-1}\|$ 自动适应模型尺度 |
| **范数计算范围** | `model.parameters()`（不含 BN buffers） | `trainable_mask` 过滤（精确控制哪些层） |
| **GPU 加速器返回值** | `(loss, model_norm)` | `(loss, model_norm, delta_norm)` |
| **首轮回退** | 使用全模型范数 | 回退到全模型范数/初始参数范数比 |

### 2.3 为什么必须改

1. **绝对范数在训练后期退化**：所有客户端从相似的全局模型出发训练，参数绝对值相近。v16 的 `model_norm` 正则对所有客户端施加相似惩罚，区分度下降。
2. **增量范数直接度量偏离**：恶意客户端的增量通常显著大于良性客户端（尤其是 Scaling Attack）。增量范数可以更精确地识别异常。
3. **量纲自适应**：ResNet-20 的 $\|\mathbf{W}\|_{\mathcal{T}} \approx 32$，因此 $\lambda_{eff} = 32.0 / 32.0 \approx 1.0$。不同模型尺度下无需手动调整 $\lambda$。

### 2.4 代码变更映射

**当前代码**（`verifl_aggregator.py` L141，GPU 路径）：
```python
loss, model_norm = self.gpu_accelerator.calculate_fitness(alpha)
cost = loss + self.lambda_reg * model_norm
return 1.0 / (cost + 1e-12)
```

**当前代码**（`gpu_accelerator.py` L110-112，范数计算）：
```python
model_norm = torch.tensor(0.0, device=self.device)
for param in self.model_template.parameters():
    model_norm += torch.sum(param ** 2)
model_norm = torch.sqrt(model_norm)
```

**目标代码**（v18f 语义，VeriflDefense 内部）：
```python
# VeriflDefense.calculate_fitness() 内部，调用 DefenseGPUContext 公共方法组合实现
loss, model_norm, delta_norm = self._calculate_fitness(alpha)  # 见下方说明
if global_prev_norm is not None and global_prev_norm > 0:
    relative_norm = delta_norm / (global_prev_norm + 1e-8)
else:
    relative_norm = model_norm / (global_prev_norm + 1e-8)  # 回退
cost = loss + self.lambda_reg * relative_norm
return 1.0 / (cost + 1e-12)
```

**DefenseGPUContext 提供的公共方法**（VeriflDefense 调用组合）：
- `flatten_clients(client_params)` → N×D GPU 矩阵
- `forward_eval(params)` → (outputs, loss)
- `compute_norm(flat_params, mask)` → L2 norm
- `flatten_params(params)` / `unflatten(flat)` → 参数形状转换

**VeriFL 特有逻辑（留在 VeriflDefense 中）**：
- `set_global_reference(global_params)` 方法：每轮调用一次，加载 $\mathbf{W}_{t-1}$ 并预算 $\|\mathbf{W}_{t-1}\|_{\mathcal{T}}$
- `_calculate_fitness(alpha)` 方法：matmul 候选聚合 + 调用 `gpu_ctx.forward_eval()` + 返回三元组 `(loss, model_norm, delta_norm)`
- `flat_trainable_mask`：用于 GPU 向量级掩码运算

### 2.5 种群初始化变更

v18f 的种群初始化相对于 v16 有一个增强：**稀疏探针个体**。

| v16 | v18f |
|-----|------|
| 1 个 FedAvg 均匀 + 全随机补齐 | 1 个 FedAvg 均匀 + **最多 4 个稀疏探针** + 随机补齐 |

稀疏探针：随机选 $k \in \{3, \min(5,N)\}$ 个客户端，仅对选中坐标赋值后归一化。目的是在早期快速探索"仅信任少数客户端"的极端方案。

---

## 3. Phase 2 差异：投影 → 裁剪

### 3.1 算法对比

**v16：全模型范数双边投影**

```
anchor_norm = ||W_anchor||_T        // 锚点的绝对模型范数
for each client i:
    client_norm = ||W_i||_T
    scale = anchor_norm / client_norm  // 可 >1（放大）或 <1（压缩）
    W̃_i = scale × W_i                // 全模型空间缩放
```

**v18f：增量空间单边裁剪**

```
d_anchor = ||W_anchor - W_{t-1}||_T  // 锚点的增量范数
for each client i:
    d_i = ||W_i - W_{t-1}||_T
    if d_i > d_anchor:
        scale = d_anchor / d_i       // 永远 ≤1（仅压缩）
        W̃_i = W_{t-1} + scale × (W_i - W_{t-1})  // 增量空间裁剪
    else:
        W̃_i = W_i                    // 保持原样
```

### 3.2 关键差异

| 维度 | v16（双边投影） | v18f（单边裁剪） |
|------|----------------|------------------|
| **操作空间** | 全模型范数 $\|\mathbf{W}_i\|$ | 增量空间 $\|\mathbf{W}_i - \mathbf{W}_{t-1}\|$ |
| **缩放方向** | 双边：$s_i$ 可 >1（放大）或 <1（压缩） | 单边：$s_i \le 1$（永远不放大） |
| **基准范数** | 锚点全模型范数 $r_a = \|\mathbf{W}_a\|$ | 锚点增量范数 $d_a = \|\boldsymbol{\Delta}_a\|$ |
| **弱更新客户端** | 被放大（$s_i > 1$）→ **噪声放大** | 保持原样 → 保留原始信噪比 |
| **公式** | $\widetilde{\mathbf{W}}_{i,\ell} = s_i \cdot \mathbf{W}_{i,\ell}$ | $\widetilde{\mathbf{W}}_{i,\ell} = \mathbf{W}_{t-1,\ell} + s_i \cdot \boldsymbol{\Delta}_{i,\ell}$ |
| **BN buffers** | trainable_mask 跳过 | trainable_mask 跳过（保留客户端原值） |
| **需要全局参考** | 否 | 是（$\mathbf{W}_{t-1}$） |
| **首轮行为** | 正常投影 | 跳过裁剪（无全局参考模型） |

### 3.3 为什么必须改

1. **双边投影的噪声放大问题**：v16 将所有客户端的模型范数统一到锚点范数。对弱更新客户端（更新量小于锚点），scale > 1 意味着放大操作。在 non-IID 场景下，弱更新客户端的小噪声会被放大。
2. **训练后期退化**：所有客户端从相似的全局模型出发，训练后期全模型范数趋同，v16 的投影退化为接近恒等映射。增量空间始终能区分"改了多少"。
3. **增量空间语义更准确**：全模型范数衡量"参数有多大"，增量范数衡量"这一轮改了多少"。对 Scaling Attack 的检测，增量空间的区分度远高于全模型空间。

### 3.4 代码变更映射

**当前代码**（`verifl_aggregator.py` L213-242）：
```python
anchor_norm = calc_l2_norm(weights_results[anchor_idx])
for client_idx in range(num_clients):
    client_norm = calc_l2_norm(weights_results[client_idx])
    scale = anchor_norm / (client_norm + 1e-9)  # 双边
    projected = [layer * scale if is_trainable else layer ...]
```

**目标代码**（v18f 语义）：
```python
# 计算锚点增量范数
def calc_delta_l2_norm(params, ref_params, trainable_mask):
    accum = 0.0
    for layer, ref, is_t in zip(params, ref_params, trainable_mask):
        if is_t:
            accum += float(np.sum((layer - ref) ** 2))
    return np.sqrt(accum)

d_anchor = calc_delta_l2_norm(weights_results[anchor_idx], global_ref, trainable_mask)

for client_idx in range(num_clients):
    d_client = calc_delta_l2_norm(weights_results[client_idx], global_ref, trainable_mask)
    if d_client > d_anchor + 1e-12:  # 单边：仅压缩超标者
        scale = d_anchor / (d_client + 1e-9)
        clipped = []
        for layer, ref, is_t in zip(weights_results[client_idx], global_ref, trainable_mask):
            if is_t:
                delta = layer - ref
                clipped.append(ref + scale * delta)  # 增量空间裁剪
            else:
                clipped.append(layer.copy())  # BN buffers 保留原值
        clipped_weights.append(clipped)
    else:
        clipped_weights.append(weights_results[client_idx])  # 未超标，保持原样
```

### 3.5 `project_to_anchor()` → `clip_to_anchor()` 重命名

当前 GPU 加速器的 `project_to_anchor()` 方法应重命名并重新实现为 `clip_to_anchor()`，反映语义变化：投影（双边）→ 裁剪（单边）。

---

## 4. Phase 3 差异：动量 BN 感知

### 4.1 对比

| 维度 | v16 | v18f |
|------|-----|------|
| 可训练参数 | $v_t = \beta v_{t-1} + \Delta_t$, $\theta_t = \theta_{t-1} + \eta v_t$ | 相同 |
| BN buffers | 同上（动量污染） | **跳过**：$v_t = 0$, $\theta_t = W_{GA}$（直接用聚合值） |
| 参数 | $\beta=0.9, \eta=0.3$ | 相同 |

### 4.2 状态

此变更已在《重生推进方案》D-3 决策中规划：

> D-3：BN-aware momentum — 保留（数学正确性修正，非公平性问题）

W-M0-1 的 Phase 3 描述中已包含此变更的实施细节。本文档不重复论证。

---

## 5. Phase 4 差异：BN 重校准

### 5.1 对比

| v16 | v18f |
|-----|------|
| Phase 3 后执行 `recalibrate_batchnorm()` | v18f 算法文档中仍保留 recal 作为后处理步骤 |

### 5.2 决策说明

《重生推进方案》D-2 决策移除 BN recalibration 的理由：
- 双重蘸取（double dipping）：val_data 同时用于 GA fitness 和 BN recal
- 基线不对称性：baselines 无 recal，对比不公平
- 学术审稿风险

v18f 算法文档（§5.4）仍保留了 recal 步骤。这是因为 v18f 文档描述的是 ShieldFL-main（Flower 框架）中的完整实现，其上下文与 FedML 不同。

**FedML 迁移决策**：遵循《重生推进方案》D-2，移除 recal。这与 v18f 的 Phase 1-3 核心算法不冲突。

---

## 6. GPU 加速器变更（→ DefenseGPUContext 重构）

> **D-6 决策**：`GPUAccelerator` 重构为共享的 `DefenseGPUContext` 工具类。公共 GPU 基础设施（模型生命周期、
> 数据生命周期、参数展平/重构、前向推理、范数计算）提取到 `DefenseGPUContext`，VeriFL 特定逻辑留在
> `VeriflDefense` 中。详见重生推进方案 D-6。

### 6.1 方法变更清单

| 方法 | v16 当前（GPUAccelerator） | v18f 目标 | 归属 | 变更 |
|------|---------|----------|------|------|
| `__init__()` | 深拷贝模型、预加载验证集、计算 trainable_mask | 同左 + 计算 `flat_trainable_mask` | `DefenseGPUContext`（公共部分）+ `VeriflDefense`（flat_trainable_mask） | **拆分** |
| `set_client_parameters()` | 展平客户端参数到 GPU 矩阵 | 相同 | `DefenseGPUContext.flatten_clients()` | **迁入共享** |
| `set_global_reference()` | **不存在** | 加载 $\mathbf{W}_{t-1}$ 到 GPU，预算 $\|\mathbf{W}_{t-1}\|_{\mathcal{T}}$ | `VeriflDefense`（VeriFL 特有） | 🔴 **新增** |
| `calculate_fitness()` | 返回 `(loss, model_norm)` | 返回 `(loss, model_norm, delta_norm)` | 🔴 **扩展** |
| `calculate_fitness()` | 返回 `(loss, model_norm)` | 返回 `(loss, model_norm, delta_norm)` | `VeriflDefense`（调用 `DefenseGPUContext` 的公共方法组合实现） | 🔴 **扩展** |
| `project_to_anchor()` | 全模型双边投影 | 废弃（逻辑迁入 VeriflDefense） | — | 🟠 **移除/重建** |
| `recalibrate_batchnorm()` | 在验证集上刷新 BN stats | 废弃 | — | 🟠 **移除**（D-2） |
| `flat_trainable_mask` | **不存在** | 布尔掩码（展平），用于 GPU 向量运算 | `VeriflDefense`（VeriFL 特有） | 🔴 **新增** |
| `load_params()` | `_load_state_from_ndarrays()` | 相同 | `DefenseGPUContext`（公共） | **迁入共享** |
| `extract_params()` | `_extract_state_to_ndarrays()` | 相同 | `DefenseGPUContext`（公共） | **迁入共享** |
| `forward_eval()` | 内嵌于 `calculate_fitness()` | 独立方法（加载参数→前向推理→返回 loss） | `DefenseGPUContext`（公共） | **提取** |
| `compute_norm()` | 内嵌于 `calculate_fitness()` | 独立方法（可选 mask） | `DefenseGPUContext`（公共） | **提取** |

### 6.2 `set_global_reference()` 规格

```python
def set_global_reference(self, global_params: List[np.ndarray]):
    """
    每轮聚合开始时调用一次。
    加载上一轮全局模型到 GPU，并预算其可训练参数 L2 范数。
    """
    flat = self._flatten(global_params)            # → GPU tensor
    self.global_prev_flat = flat
    masked = flat[self.flat_trainable_mask]
    self.global_prev_norm = torch.sqrt(torch.sum(masked ** 2)).item()
```

### 6.3 `calculate_fitness()` 扩展

```python
def calculate_fitness(self, alpha):
    # ... 现有：矩阵乘法聚合 + 前向推理得到 loss + 全模型 model_norm

    # 新增：增量范数
    if self.global_prev_flat is not None:
        delta_flat = aggregated_flat - self.global_prev_flat
        delta_masked = delta_flat[self.flat_trainable_mask]
        delta_norm = torch.sqrt(torch.sum(delta_masked ** 2)).item()
    else:
        delta_norm = float('nan')

    return float(loss), float(model_norm), float(delta_norm)
```

---

## 7. 超参数映射

### 7.1 变更超参数

| 参数 | v16 值 | v18f 值 | 变更原因 |
|------|--------|---------|---------|
| `lambda_reg` | 0.1 | **32.0** | 适配相对范数：$\lambda_{eff} = 32.0 / \|\mathbf{W}\| \approx 1.0$（ResNet-20） |

### 7.2 不变超参数

| 参数 | 值 | 说明 |
|------|---|------|
| `pop_size` | 15 | 种群大小 |
| `generations` | 10 | 进化代数 |
| `server_momentum` | 0.9 | 动量系数 β |
| `server_lr` | 0.3 | 服务端学习率 η |
| `mutation_prob` | 0.1 | 变异概率 |
| `mutation_sigma` | 0.05 | 变异标准差 |
| `tournament_k` | 2 | 锦标赛大小 |

### 7.3 新增常量

| 常量 | 值 | 用途 |
|------|---|------|
| `ε_f` (eps_fitness) | $10^{-12}$ | 适应度分母稳定项 |
| `ε_r` (eps_relative) | $10^{-8}$ | 相对范数分母稳定项 |
| `ε_p` (eps_projection) | $10^{-9}$ | Phase 2 缩放分母稳定项 |
| `ε_c` (eps_clip) | $10^{-12}$ | Phase 2 裁剪判定阈值 |
| `ε_n` (eps_normalize) | $10^{-9}$ | 归一化分母稳定项 |

### 7.4 λ 的量纲自适应解释

v18f 的 `lambda_reg=32.0` 看起来远大于 v16 的 `0.1`，但实际正则力度相当：

$$
\lambda_{eff} = \frac{\lambda_f}{\|\mathbf{W}_{t-1}\|_{\mathcal{T}} + \varepsilon_r} \approx \frac{32.0}{32.0} \approx 1.0 \quad (\text{ResNet-20})
$$

而 v16 的 `0.1` 直接作用于绝对模型范数（$\|\mathbf{W}\| \approx 32$），正则贡献为 $0.1 \times 32 = 3.2$。

v18f 的正则贡献为 $32.0 \times r(\alpha)$，其中 $r(\alpha) = \|\Delta\| / \|\mathbf{W}_{t-1}\|$。正常训练时 $r \approx 0.01$∼$0.1$，正则贡献 $\approx 0.32$∼$3.2$。

两者在正常训练范围内量级相当，但 v18f 对 Scaling Attack（$r \gg 1$）的惩罚力度远强于 v16。

---

## 8. 代码级变更清单

### 8.1 文件变更矩阵

以下将变更映射到重生推进方案的 W-M0-1 工作项中：

| # | 文件 | 变更类型 | 涉及阶段 | 描述 |
|---|------|---------|---------|------|
| C-1 | `verifl_defense.py` (新建) | 🔴 新建 | All | 实现 v18f 三阶段算法（不是简单迁移 v16 逻辑） |
| C-2 | `defense_gpu_context.py` (新建) + `verifl_defense.py` | 🔴 重构 | Phase 1 | 从 `gpu_accelerator.py` 提取公共 GPU 基础设施为 `DefenseGPUContext`；VeriFL 特有的 `set_global_reference()`、`calculate_fitness()` 三元组、`flat_trainable_mask` 留在 `VeriflDefense` 中 |
| C-3 | `gpu_accelerator.py` → `.deprecated` | 🟠 废弃 | Phase 2, 4 | 原文件整体废弃（已被 `defense_gpu_context.py` + `VeriflDefense` 替代），`project_to_anchor()` 和 `recalibrate_batchnorm()` 不再存在 |
| C-4 | `verifl_defense.py` | 🔴 新增 | Phase 1 | 实现 v18f 适应度函数（relative delta-norm） |
| C-5 | `verifl_defense.py` | 🔴 新增 | Phase 2 | 实现 delta-space 单边裁剪（替代双边投影） |
| C-6 | `verifl_defense.py` | 🟠 修复 | Phase 3 | BN-aware momentum（trainable_mask 门控） |
| C-7 | `verifl_defense.py` | 🔴 新增 | Phase 1 | 稀疏探针种群初始化 |
| C-8 | config/YAML | 🟡 更新 | — | `lambda_reg` 默认值从 0.1 → 32.0 |

### 8.2 `defend_on_aggregation` 完整流程（v18f）

```
输入: raw_client_grad_list = [(sample_num_0, state_dict_0), ...]

0. 提取参数列表
   weights_results = [extract_state_dict(x) for x in raw_client_grad_list]
   global_ref = self.global_model_buffer  # 上一轮全局模型（首轮为 None）

1. GPU 预加载
   gpu.set_client_parameters(weights_results)
   if global_ref is not None:
       gpu.set_global_reference(global_ref)
       global_prev_norm = gpu.global_prev_norm

2. Phase 1: GA 搜索 + Relative Delta-Norm
   population = init_population(N)  # 含 FedAvg + 稀疏探针 + 随机
   for g in range(generations):
       for alpha in population:
           loss, model_norm, delta_norm = gpu.calculate_fitness(alpha)
           if global_ref is not None:
               relative_norm = delta_norm / (global_prev_norm + 1e-8)
           else:
               relative_norm = model_norm / (global_prev_norm + 1e-8)  # 回退
           cost = loss + lambda_reg * relative_norm
           fitness = 1.0 / (cost + 1e-12)
       # ... 遗传操作（与 v16 完全相同）
   输出: alpha_star

3. Phase 2: Delta-Space 单边裁剪
   anchor_idx = argmax(alpha_star)
   d_anchor = calc_delta_l2_norm(weights_results[anchor_idx], global_ref)
   for each client i:
       d_i = calc_delta_l2_norm(weights_results[i], global_ref)
       if d_i > d_anchor + 1e-12:
           scale = d_anchor / (d_i + 1e-9)
           clipped_i = [ref + scale * (W - ref) if trainable else W.copy()]
       else:
           clipped_i = weights_results[i]  # 保持原样

4. 融合
   W_GA = aggregate_weighted(clipped_weights, alpha_star)

5. Phase 3: BN-aware Momentum
   for each layer l:
       if trainable_mask[l]:
           delta = W_GA[l] - global_ref[l]
           velocity[l] = beta * velocity[l] + delta
           W_final[l] = global_ref[l] + eta * velocity[l]
       else:  # BN buffers
           velocity[l] = 0
           W_final[l] = W_GA[l]  # 直接用 GA α*-加权聚合值

6. 更新跨轮缓冲
   self.global_model_buffer = deepcopy(W_final)
   self.velocity_buffer = velocity

7. 返回 OrderedDict（W_final）
```

---

## 9. 实施优先级与依赖关系

### 9.1 依赖图

```
C-2 (DefenseGPUContext 提取 + VeriFL GPU 逻辑迁入)
  ├── C-4 (Phase 1 适应度函数) → 依赖 C-2 的 delta_norm 返回值
  └── C-5 (Phase 2 单边裁剪) → 依赖 global_ref
C-3 (废弃旧 gpu_accelerator.py) → 在 C-5 完成后可安全废弃
C-6 (Phase 3 BN-aware momentum) → 独立于 C-2/C-4/C-5
C-7 (稀疏探针) → 独立
C-8 (lambda_reg 默认值) → 最后
```

### 9.2 建议实施顺序

1. **C-2**：DefenseGPUContext 提取 + VeriFL 特定 GPU 逻辑迁入 VeriflDefense
2. **C-7**：稀疏探针种群初始化（独立模块）
3. **C-4**：Phase 1 适应度函数（依赖 C-2）
4. **C-5**：Phase 2 单边裁剪（依赖 C-2 的 global_ref）
5. **C-6**：Phase 3 BN-aware momentum
6. **C-3**：废弃旧 gpu_accelerator.py（C-2/C-4/C-5 完成后安全执行）
7. **C-8**：更新默认超参数
8. **C-1**：整合为完整的 `verifl_defense.py`（C-2∼C-7 的组装）

---

## 10. 与重生推进方案的关系

### 10.1 对 W-M0-1 的影响

重生推进方案 W-M0-1 的原始描述是：

> "GA 搜索逻辑保持不变，直接迁移"

**这在 v18f 下不正确**。正确描述应为：

> VeriflDefense 的实现应遵循 v18f 算法规格，不是简单迁移 v16 逻辑。具体变更：
> - Phase 1 适应度函数改为 relative delta-norm（C-4）
> - Phase 2 从双边投影改为单边裁剪（C-5）
> - Phase 3 BN-aware momentum（原 D-3 决策不变）
> - BN recalibration 移除（原 D-2 决策不变）
> - 种群初始化增加稀疏探针（C-7）
> - GPU 加速器重构为 DefenseGPUContext（C-2），VeriFL 特有逻辑不再是"搬运"而是基于共享工具类重新实现

### 10.2 对其他工作项的影响

| 工作项 | 影响 | 说明 |
|--------|------|------|
| W-M0-2（FedMLDefender 注册） | 无影响 | 注册机制不涉及内部算法 |
| W-M0-3（ShieldFLAggregator） | 无影响 | 聚合器不包含防御逻辑 |
| W-M0-4（main 简化） | 无影响 | 入口点不涉及算法细节 |
| W-M0-5（run_experiment.sh） | **需更新** | `lambda_reg` 默认值变更，YAML 需反映 |
| W-M0-6（重构 gpu_accelerator → DefenseGPUContext） | **变更** | 不再是简单删除方法，而是整体重构为共享工具类 + VeriFL 特有逻辑分离（D-6 决策） |
| D-3（BN-aware momentum） | 不变 | v18f 与 D-3 一致 |

### 10.3 新增验收标准

| AC 编号 | 验收标准 | 验证方法 |
|---------|---------|---------|
| AC-v18f-1 | VeriflDefense 的 `calculate_fitness()` 返回三元组 `(loss, model_norm, delta_norm)`（基于 DefenseGPUContext 公共方法组合） | 单元测试 |
| AC-v18f-2 | 适应度函数使用 relative delta-norm（非绝对模型范数） | 代码审计 + telemetry 验证 |
| AC-v18f-3 | Phase 2 为单边裁剪（scale ≤ 1），不放大弱更新 | 日志检查：无 scale > 1 |
| AC-v18f-4 | Phase 2 在增量空间操作（$\mathbf{W}_{t-1} + s \cdot \Delta$，非 $s \cdot \mathbf{W}$） | 代码审计 |
| AC-v18f-5 | `lambda_reg` 默认值为 32.0 | 配置检查 |
| AC-v18f-6 | 种群初始化含稀疏探针个体 | 日志 / 单元测试 |

---

## 11. v16→v18f 演化脉络

供参考，完整版本演化链（来自 v18f 算法文档 §17）：

| 版本 | 核心特性 | 问题 |
|------|---------|------|
| **v16** | 三阶段：GA + 全模型双边投影 + 全参数动量 | BN 动量污染；双边投影后期退化；全模型范数尺度敏感 |
| **v18a** | 修复 BN 动量 bug；Phase 2 改为增量空间裁剪 | Phase 1 仍用绝对范数 |
| **v18b** | Phase 1 改为增量范数正则 $\lambda=1.0$ | λ 固定值不自适应 |
| **v18c** | 尝试去除范数正则 | 防御失效 → 废弃 |
| **v18d** | 尝试秩融合替代加权平均 | 不收敛 → 废弃 |
| **v18f** | 相对增量范数 $\lambda_f=32.0$；量纲自适应 | **当前活跃版本** |

FedML 迁移直接跳到 v18f 终态，无需经过中间版本。

---

> **文档版本**：v1.0  
> **创建日期**：基于 algrothm.md（v18f 完整算法说明）与 FedML VeriFL 实现代码比对生成  
> **关联文档**：重生推进方案.md § W-M0-1
