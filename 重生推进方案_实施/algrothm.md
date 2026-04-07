# VeriFL-v18f 完整算法说明文档

> **文档性质**：面向论文撰写 Agent 的完备技术参考文档  
> **归属项目**：ShieldFL — 基于 Flower 框架的拜占庭鲁棒联邦学习仿真系统  
> **策略全名**：VeriFL-v18f-Relative: Validation-Driven Three-Phase Defense with Relative Delta-Norm  
> **代码位置**：`src/strategies/ours/v18f_relative.py`  
> **最后更新**：2026-04-06

---

## 目录

1. [研究背景与动机](#1-研究背景与动机)
2. [算法定位与设计哲学](#2-算法定位与设计哲学)
3. [系统架构与算法嵌入点](#3-系统架构与算法嵌入点)
4. [符号与记号约定](#4-符号与记号约定)
5. [核心算法：三阶段防御流水线](#5-核心算法三阶段防御流水线)
   - 5.1 [Phase 1: GA 零阶搜索 + Relative Delta-Norm 正则（侦查）](#51-phase-1-ga-零阶搜索--relative-delta-norm-正则侦查)
   - 5.2 [Phase 2: Delta-Space 单边裁剪（去势）](#52-phase-2-delta-space-单边裁剪去势)
   - 5.3 [Phase 3: BN 感知的全局动量平滑（平滑）](#53-phase-3-bn-感知的全局动量平滑平滑)
   - 5.4 [后处理: BatchNorm 重校准](#54-后处理-batchnorm-重校准)
6. [完整伪代码](#6-完整伪代码)
7. [超参数完整字典](#7-超参数完整字典)
8. [适应度函数详解](#8-适应度函数详解)
9. [GPU 加速架构](#9-gpu-加速架构)
10. [数据流全景](#10-数据流全景)
11. [威胁模型与防御目标](#11-威胁模型与防御目标)
12. [对各类攻击的防御机制分析](#12-对各类攻击的防御机制分析)
13. [与经典方法的关键区别](#13-与经典方法的关键区别)
14. [继承体系与代码架构](#14-继承体系与代码架构)
15. [实验环境设定](#15-实验环境设定)
16. [隐含假设与已知局限](#16-隐含假设与已知局限)
17. [从 v16 到 v18f 的演化脉络](#17-从-v16-到-v18f-的演化脉络)

---

## 1. 研究背景与动机

### 1.1 联邦学习面临的拜占庭威胁

联邦学习（Federated Learning, FL）允许多个客户端在不共享原始数据的前提下协作训练全局模型。然而，分布式场景天然面临**拜占庭威胁**：部分客户端可能因被入侵、数据污染或恶意参与，上传损害全局模型的参数更新。

经典攻击模式包括：

| 攻击类型 | 攻击方式 | 目标 |
|---------|---------|------|
| **Byzantine Attack** | 上传随机噪声或对抗性梯度 | 破坏模型收敛 |
| **Label Flip Attack** | 翻转训练标签（如 $y \to 9-y$） | 降低特定类别准确率 |
| **Scaling Attack** | 正常训练后将梯度按倍数放大 | 操纵聚合结果的方向 |
| **Backdoor Attack** | 在训练数据中植入触发器模式 | 使模型在特定输入条件下输出指定标签 |
| **Model Replacement** | 放大后门更新以覆盖其他客户端 | 使全局模型完全被后门控制 |

### 1.2 现有防御的不足

现有服务端聚合防御可大致分为两类：

1. **统计鲁棒聚合**（如 Median、Trimmed Mean、Multi-Krum、Bulyan）：基于坐标级或几何距离的异常剔除。这些方法在面对自适应攻击（如 Scaling 或定向对齐攻击）时缺乏鲁棒性，因为攻击者可以精心设计更新使其在统计特征上与良性更新无法区分。

2. **信任校准聚合**（如 FLTrust, Zeno）：依赖服务端可信数据集/先验知识进行梯度筛选。这些方法对可信数据的质量和规模敏感，且本质上是分类式（接受/拒绝）而非连续优化。

### 1.3 VeriFL-v18f 的核心创新

VeriFL-v18f 提出了一种全新的范式：**将服务端聚合视为一个优化问题**，利用微型遗传算法（Micro-GA）在权重单纯形上搜索最优客户端组合权重，并通过三阶段协同防御实现多层次鲁棒性：

- **GA 搜索 + Relative Delta-Norm 正则**替代了手工规则，让验证集损失和模型变化量共同驱动权重分配，且正则强度自动适应模型尺度；
- **Delta-Space 单边裁剪**在增量空间中约束更新幅度，仅裁剪超标更新、不放大弱更新，消除 Scaling Attack 的放大效应；
- **BN 感知的服务端动量**引入跨轮记忆与惯性，同时正确区分可训练参数与 BN 缓冲区。

这种"搜索 + 裁剪 + 平滑"的三阶段设计使算法可以同时应对多种异构攻击。

### 1.4 v18f 相对于 v16 的核心改进

v18f 是从 v16 → v18a → v18b → v18f 一路迭代的产物。与 v16 相比，三个阶段均有实质性改动：

| 阶段 | v16 | v18f | 改进动机 |
|------|-----|------|---------|
| Phase 1 正则项 | 绝对模型范数 $\|\mathbf{W}\|_2$ | 相对增量范数 $\|\mathbf{\Delta}\|_2 / \|\mathbf{W}_{t-1}\|_2$ | 消除对模型尺度的依赖；直接度量"变化量"而非"绝对大小" |
| Phase 1 λ | 0.1 | 32.0 | 量纲统一后需要更大系数才能产生等价惩罚力度 |
| Phase 2 空间 | 全模型范数投影（双边缩放） | 增量空间单边裁剪 | 避免"放大弱更新、缩小强更新"引入的信噪比失真 |
| Phase 3 BN | 对所有参数施加动量 | BN buffers 跳过动量 + recalibration 覆盖 | 修复 BN running stats 被动量污染的 bug |

---

## 2. 算法定位与设计哲学

### 2.1 在联邦学习系统中的定位

VeriFL-v18f 运行在**服务器端聚合阶段**，直接接管 Flower 框架的 `aggregate_fit` 接口。每一轮联邦训练中，当服务器收到所有参与客户端上传的模型参数后，VeriFL-v18f 取代标准 FedAvg 的简单加权平均，执行三阶段防御流水线。

```
客户端本地训练 → 上传参数 → [VeriFL-v18f 聚合] → 下发全局模型 → 下一轮
                              ↑ 核心算法在此处
```

### 2.2 设计哲学："侦查 → 去势 → 平滑"

VeriFL-v18f 的设计遵循**纵深防御**（Defense in Depth）思想：

| 阶段 | 代号 | 核心思想 | 防御目标 |
|------|------|---------|---------|
| Phase 1 | 侦查 | 通过零阶优化在权重空间搜索，利用验证损失 + 相对增量范数自动识别并降权恶意客户端 | 任意偏离全局方向的攻击 |
| Phase 2 | 去势 | 在增量空间中将超标客户端裁剪到锚点的增量半径以内（仅裁不放） | Scaling Attack、Model Replacement |
| Phase 3 | 平滑 | 在全局模型层面应用 BN 感知的动量平滑，使训练轨迹抵抗单轮扰动 | 后期震荡、间歇性攻击 |

三阶段的关键特性在于：每一阶段解决不同维度的问题，且前一阶段为后一阶段提供更高质量的输入。

### 2.3 关键设计决策

1. **无历史注入的种群初始化**：每轮 GA 搜索从零开始初始化种群，不继承上一轮的最优解。这避免了跨轮的记忆偏差——若某轮被攻击者误导产生了错误的最优权重，这一错误不会被下一轮继承。跨轮连续性由 Phase 3 的动量机制单独负责。

2. **零阶优化而非梯度方法**：GA 不需要对聚合目标求梯度，因此可以处理复杂的、非可微的适应度函数（如带 NaN 保护的验证损失倒数）。

3. **锚点而非中位数**：选择 GA 最信任的客户端作为锚点，而非统计中位数。这意味着锚点的选择本身就经过了验证集驱动的优化，而不是简单的几何中心。

4. **增量空间而非全模型空间**：Phase 2 的裁剪在 $\mathbf{W}_i - \mathbf{W}_{t-1}$（增量）空间中执行，而非对 $\mathbf{W}_i$（全模型）做投影。这使得裁剪操作直接度量"这个客户端改了多少"，与训练动态的语义更对齐。

5. **单边裁剪而非双边缩放**：只裁剪增量超标的客户端（向下压缩），不对增量偏小的客户端做放大。这避免了 v16 双边投影（Bilateral Projection）中"放大弱更新引入噪声"的问题。

6. **相对正则而非绝对正则**：将增量范数除以全局模型范数，使得正则项对模型尺度具有不变性。无论模型参数量大小或训练阶段（参数从初始化到收敛的量级变化），正则力度都是稳定的。

7. **BN 感知的动量**：Phase 3 区分可训练参数和 BN 缓冲区，仅对前者施加动量。BN buffers 使用 GA 聚合的直接输出，然后通过 recalibration 重新校准。

---

## 3. 系统架构与算法嵌入点

### 3.1 整体系统流程

```
[命令行入口 src/main.py]
    │
    ├─ 解析命令行参数 (--scenario, --method)
    ├─ 合并配置文件 (merge_configs)
    ├─ 创建 Config 数据类
    │
    ▼
[SimulationRunner (src/core/runner.py)]
    │
    ├─ 1. 数据准备 (DataFactory)
    │     ├─ 加载 CIFAR-10 训练集 + 测试集
    │     ├─ 从训练集中切分服务端验证集 (val_ratio=500 样本)
    │     ├─ 从训练集中切分可信数据集 (trust_ratio=500 样本)
    │     ├─ 剩余训练数据按 Dirichlet(α=0.5) 分配给各客户端
    │     └─ 数据互斥断言: val ∩ trust ∩ clients = ∅
    │
    ├─ 2. 模型创建 (ModelFactory)
    │     └─ 实例化 ResNet20
    │
    ├─ 3. 攻击配置 (AttackManager)
    │     ├─ 前 num_malicious 个客户端标记为恶意
    │     └─ 配置攻击策略(label_flip/backdoor/scaling/byzantine)
    │
    ├─ 4. 评估器初始化 (Evaluator)
    │     ├─ get_eval_fn() → 用于 GA 适应度计算 (在 val_loader 上)
    │     ├─ evaluate_test() → 每轮泛化评估 (在 test_loader 上)
    │     └─ evaluate_backdoor_test() → ASR 计算
    │
    ├─ 5. 策略构建 (StrategyBuilder)
    │     ├─ 实例化 StrategyV18FRelative
    │     ├─ 初始化 GPU 加速器 (model_template + val_data)
    │     └─ 包装 BN 适配层 (_wrap_with_bn_adaptation)
    │
    ├─ 6. 客户端工厂 (client_fn)
    │     ├─ BenignClient: SGD(lr=0.01, momentum=0.9)
    │     └─ MaliciousClient: 委托 AttackManager.execute()
    │
    └─ 7. 启动 Flower 仿真
          └─ fl.simulation.start_simulation()
                │
                └─ 每轮: 客户端训练 → aggregate_fit(VeriFL-v18f) → evaluate_fn
```

### 3.2 aggregate_fit 的调用上下文

在 Flower 框架中，`aggregate_fit` 是策略的核心接口，每轮由服务器自动调用：

```python
# Flower 框架内部伪代码
for round in range(1, num_rounds + 1):
    selected_clients = strategy.configure_fit(round, ...)
    results, failures = run_clients_in_parallel(selected_clients)
    parameters, metrics = strategy.aggregate_fit(round, results, failures)
    loss, eval_metrics = strategy.evaluate(round, parameters)
```

`results` 的数据结构为 `List[Tuple[ClientProxy, FitRes]]`，其中每个 `FitRes` 包含该客户端训练后的完整模型参数（全 `state_dict`，含 BN buffers）。

### 3.3 类继承关系

```
flwr.server.strategy.FedAvg          ← Flower 官方基类
    └── MicroGABase                  ← GA 基座层 (src/strategies/ours/ga_base.py)
        └── StrategyV18FRelative     ← v18f 策略 (src/strategies/ours/v18f_relative.py)
```

- **FedAvg** 提供：Flower 框架接口兼容性（`configure_fit`、`configure_evaluate` 等）
- **MicroGABase** 提供：基础遗传操作（锦标赛选择、交叉、变异）、GPU 加速器管理、telemetry 记录
- **StrategyV18FRelative** 实现：三阶段防御的完整逻辑，完全覆盖 `aggregate_fit`、`calculate_fitness`、`_init_population`、`_compute_fitness_details`

---

## 4. 符号与记号约定

| 符号 | 含义 |
|------|------|
| $N$ | 当前轮次参与的客户端总数 |
| $\mathbf{W}_i = \{\mathbf{W}_{i,\ell}\}_{\ell=1}^{L}$ | 第 $i$ 个客户端上传的完整模型参数（$L$ 层，含 BN） |
| $\mathbf{W}_{t-1}$ | 上一轮全局模型参数（`global_model_buffer`） |
| $\boldsymbol{\alpha} \in \Delta^{N-1}$ | 聚合权重向量，位于 (N-1)-单纯形上 |
| $\Delta^{N-1} = \left\{\boldsymbol{\alpha} \in \mathbb{R}^N \mid \alpha_i \ge 0,\; \sum_{i=1}^N \alpha_i = 1\right\}$ | (N-1)-概率单纯形 |
| $\mathcal{A}(\{\mathbf{W}_i\}, \boldsymbol{\alpha}) = \sum_{i=1}^N \alpha_i \mathbf{W}_i$ | 加权聚合算子 |
| $L_{val}(\mathbf{W})$ | 在服务端验证集上的交叉熵损失函数值 |
| $\boldsymbol{\Delta}_i = \mathbf{W}_i - \mathbf{W}_{t-1}$ | 客户端 $i$ 的增量（与上一轮全局模型的差值） |
| $\boldsymbol{\Delta}(\boldsymbol{\alpha}) = \mathcal{A}(\{\mathbf{W}_i\}, \boldsymbol{\alpha}) - \mathbf{W}_{t-1}$ | 候选聚合模型的增量 |
| $\lambda_f$ | 相对增量范数惩罚系数（`lambda_reg`，默认 32.0） |
| $\mathcal{T}$ | 可训练参数层索引集（由 `trainable_mask` 定义） |
| $\|\cdot\|_{\mathcal{T}}$ | 仅在可训练层上计算的 $L_2$ 范数 |
| $\mathcal{F}(\boldsymbol{\alpha})$ | 适应度函数 |
| $P$ | GA 种群大小（`pop_size`，默认 15） |
| $G$ | GA 进化代数（`generations`，默认 10） |
| $\boldsymbol{\alpha}^*$ | GA 搜索得到的最优权重向量 |
| $a = \arg\max_i \alpha_i^*$ | 锚点客户端索引 |
| $d_a = \|\boldsymbol{\Delta}_a\|_{\mathcal{T}}$ | 锚点的增量范数 |
| $\widetilde{\mathbf{W}}_i$ | 裁剪后的客户端参数 |
| $\mathbf{W}_{GA}$ | 裁剪后加权融合的聚合参数 |
| $\mathbf{W}_t$ | 第 $t$ 轮的全局模型参数（Phase 3 输出） |
| $\mathbf{V}_t$ | 第 $t$ 轮的速度缓冲区 |
| $\beta$ | 服务端动量系数（`server_momentum`，默认 0.9） |
| $\eta$ | 服务端学习率（`server_lr`，默认 0.3） |
| $\varepsilon_f = 10^{-12}$ | 适应度分母稳定项 |
| $\varepsilon_r = 10^{-8}$ | 相对范数分母稳定项 |
| $\varepsilon_p = 10^{-9}$ | 裁剪缩放分母稳定项 |

---

## 5. 核心算法：三阶段防御流水线

在每轮聚合中，算法接收客户端参数集合 $\{\mathbf{W}_i\}_{i=1}^N$，经过以下严格有序的步骤，输出新的全局模型参数 $\mathbf{W}_t$：

$$
\{\mathbf{W}_i\}_{i=1}^N \xrightarrow[\text{Phase 1}]{\text{GA 搜索}} \boldsymbol{\alpha}^* \xrightarrow[\text{Phase 2}]{\text{Delta 裁剪}} \{\widetilde{\mathbf{W}}_i\}_{i=1}^N \xrightarrow[\text{融合}]{\mathcal{A}} \mathbf{W}_{GA} \xrightarrow[\text{Phase 3}]{\text{BN 感知动量}} \mathbf{W}_t
$$

### 5.1 Phase 1: GA 零阶搜索 + Relative Delta-Norm 正则（侦查）

**目标**：在客户端权重的概率单纯形 $\Delta^{N-1}$ 上搜索最优聚合权重 $\boldsymbol{\alpha}^*$，使得聚合后模型在服务端验证集上的表现最优，同时惩罚偏离全局模型过大的候选解。

**核心思想**：恶意客户端的参数通常导致聚合模型在验证集上的损失爆炸或增量过大。通过 GA 搜索，恶意客户端自然被分配到极低的权重。

#### 5.1.1 种群初始化

在每轮聚合开始时，从零初始化一个由 $P$ 个个体（权重向量）构成的种群 $\mathcal{P}^{(0)} = \{\boldsymbol{\alpha}_j^{(0)}\}_{j=1}^P$：

1. **FedAvg 基准个体**（1 个）：
$$
\boldsymbol{\alpha}_1 = \frac{1}{N}\mathbf{1}
$$
确保 GA 搜索不会劣于无防御基线。

2. **稀疏探针个体**（最多 4 个）：每个探针随机选择 $k$ 个客户端（$k \sim \text{Uniform}\{3, \min(5, N)\}$，当 $N < 3$ 时 $k = N$），在选中坐标赋值 1 后归一化：
$$
\boldsymbol{\alpha}_{\text{probe}} = \frac{\mathbf{e}_{S}}{\|\mathbf{e}_{S}\|_1}, \quad S \subseteq \{1, \ldots, N\},\; |S| = k
$$
**意义**：在单纯形上注入"低支持度"解，使 GA 在早期就能快速探索"仅信任少数客户端"的极端方案。

3. **随机个体**（补齐至 $P$ 个）：从非负均匀分布采样后归一化：
$$
\alpha_i \sim U(0,1), \quad \boldsymbol{\alpha} \leftarrow \boldsymbol{\alpha} / \|\boldsymbol{\alpha}\|_1
$$

**关键设计决策**：**不注入历史轮次的最优权重**。这是经过 v1-v15 迭代验证的设计选择——历史注入会让 GA 在面对攻击模式变化时产生滞后性记忆偏差。跨轮连续性完全由 Phase 3 的动量机制负责。

#### 5.1.2 适应度函数（v18f 核心改动）

对任一候选权重向量 $\boldsymbol{\alpha}$：

**Step 1**: 构造候选全局模型：
$$
\mathbf{W}(\boldsymbol{\alpha}) = \sum_{i=1}^N \alpha_i \mathbf{W}_i = \mathcal{A}(\{\mathbf{W}_i\}, \boldsymbol{\alpha})
$$

**Step 2**: 在服务端验证集上评估交叉熵损失：
$$
L_{val} = L_{val}(\mathbf{W}(\boldsymbol{\alpha}))
$$
若 $L_{val}$ 为 NaN 或 Inf，直接返回 $\mathcal{F} = 0$。

**Step 3**: 计算候选模型与上一轮全局模型之间的**增量范数**（仅可训练参数）：
$$
\|\boldsymbol{\Delta}(\boldsymbol{\alpha})\|_{\mathcal{T}} = \sqrt{\sum_{\ell \in \mathcal{T}} \|\mathbf{W}_\ell(\boldsymbol{\alpha}) - \mathbf{W}_{t-1,\ell}\|_F^2}
$$

**Step 4**: 计算上一轮全局模型的范数（仅可训练参数）：
$$
\|\mathbf{W}_{t-1}\|_{\mathcal{T}} = \sqrt{\sum_{\ell \in \mathcal{T}} \|\mathbf{W}_{t-1,\ell}\|_F^2}
$$

**Step 5**: 计算**相对增量范数**：
$$
r(\boldsymbol{\alpha}) = \frac{\|\boldsymbol{\Delta}(\boldsymbol{\alpha})\|_{\mathcal{T}}}{\|\mathbf{W}_{t-1}\|_{\mathcal{T}} + \varepsilon_r}
$$

**Step 6**: 计算代价函数与适应度：
$$
\mathcal{C}(\boldsymbol{\alpha}) = L_{val}(\mathbf{W}(\boldsymbol{\alpha})) + \lambda_f \cdot r(\boldsymbol{\alpha})
$$
$$
\mathcal{F}(\boldsymbol{\alpha}) = \frac{1}{\mathcal{C}(\boldsymbol{\alpha}) + \varepsilon_f}
$$

**回退机制**：若增量范数不可用（首轮 $\mathbf{W}_{t-1}$ 为空或计算异常），退化为全模型范数：
$$
r_{\text{fallback}}(\boldsymbol{\alpha}) = \frac{\|\mathbf{W}(\boldsymbol{\alpha})\|_{\mathcal{T}}}{\|\mathbf{W}_{t-1}\|_{\mathcal{T}} + \varepsilon_r}
$$

#### 5.1.3 相对正则项的数学性质

**量纲不变性**：在同一轮内，$\|\mathbf{W}_{t-1}\|_{\mathcal{T}}$ 为常数，因此：

$$
\mathcal{C}(\boldsymbol{\alpha}) = L_{val} + \frac{\lambda_f}{\|\mathbf{W}_{t-1}\|_{\mathcal{T}} + \varepsilon_r} \cdot \|\boldsymbol{\Delta}(\boldsymbol{\alpha})\|_{\mathcal{T}}
$$

定义**有效正则系数**：
$$
\lambda_{\text{eff}} = \frac{\lambda_f}{\|\mathbf{W}_{t-1}\|_{\mathcal{T}} + \varepsilon_r}
$$

对于 ResNet-20（$\|\mathbf{W}\|_{\mathcal{T}} \approx 32$），$\lambda_{\text{eff}} \approx 32.0 / 32.0 = 1.0$。

**与 v16 的等价关系**：v18f 的相对正则在每一轮内等价于 v16 的绝对增量范数正则，且 $\lambda_{\text{eff}}$ 自动随模型尺度调整。GA 内部的排序不受影响（同一轮 $\|\mathbf{W}_{t-1}\|$ 为常数，不改变排序）。跨轮看，$\lambda_{\text{eff}}$ 会随训练进程自适应——训练后期模型范数变大时，正则力度自动下降；反之上升。

**为什么用增量范数而非全模型范数**：v16 使用全模型范数 $\|\mathbf{W}(\boldsymbol{\alpha})\|_2$。但全模型范数惩罚的是参数的"绝对大小"，这会偏向选择"参数值小"的客户端，而非"变化合理"的客户端。v18f 用增量范数，直接度量"候选聚合结果相对于上一轮全局模型改变了多少"，语义更准确。

#### 5.1.4 遗传算子

每一代进化执行以下操作（与 v16 完全相同的遗传操作框架）：

**(a) 评估与精英保留**

评估当前种群中所有个体的适应度，维护全局历史最优个体 $\boldsymbol{\alpha}^*$ 及其适应度 $\mathcal{F}^*$。精英保留：每代新种群的第一个位置始终为 $\boldsymbol{\alpha}^*$。

**(b) 锦标赛选择**（Tournament Size $k = 2$）

从当前种群中有放回地随机选择 2 个个体，取适应度较高者作为一个父代。重复此过程生成与种群等大的父代池：
$$
\text{parent} = \arg\max(\mathcal{F}(\boldsymbol{\alpha}_{i_1}), \mathcal{F}(\boldsymbol{\alpha}_{i_2})), \quad i_1, i_2 \sim \text{Uniform}(\{1,\ldots,P\})
$$

**(c) 线性交叉**

对两个父代 $\mathbf{p}_1, \mathbf{p}_2$，生成子代：
$$
\mathbf{c} = \beta \cdot \mathbf{p}_1 + (1 - \beta) \cdot \mathbf{p}_2, \quad \beta \sim U(0, 1)
$$
随后取绝对值并归一化：
$$
\mathbf{c} \leftarrow \frac{|\mathbf{c}|}{\sum_i |c_i| + \varepsilon_n}, \quad \varepsilon_n = 10^{-9}
$$

**(d) 高斯变异**（概率 $p_m = 0.1$，标准差 $\sigma = 0.05$）

以概率 $p_m$ 对子代施加扰动：
$$
\mathbf{c} \leftarrow \mathbf{c} + \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})
$$
随后取绝对值并归一化。

**归一化退化保护**：若归一化分母 $< 10^{-9}$，退化为均匀分布 $\boldsymbol{\alpha} = \frac{1}{N}\mathbf{1}$。

#### 5.1.5 停滞检测与回退机制

- **代级回退**：若某代中 $\boldsymbol{\alpha}^*$ 为空或 $\mathcal{F}^* \le 0$（所有候选解均异常），则该代直接丢弃种群并重新初始化，不执行遗传操作。
- **全局回退**：若经过 $G$ 代迭代后仍无有效最优解，则回退到 FedAvg 均匀权重 $\boldsymbol{\alpha}^* = \frac{1}{N}\mathbf{1}$。

#### 5.1.6 Phase 1 的计算复杂度

每轮 GA 搜索需要 $P \times G$ 次适应度评估。每次评估包含：
- 加权聚合：$O(N \times D)$（GPU 矩阵乘法加速）
- 增量范数计算：$O(D)$（可训练参数上，GPU 向量运算）
- 全局模型范数计算：$O(D)$（每轮仅计算一次，跨候选解共享）
- 前向推理：一次完整的模型前向传播（在 GPU 上，使用预加载的验证集）

默认配置 $P = 15, G = 10$ 意味着每轮 150 次适应度评估。GPU 加速器通过将所有客户端参数展平为矩阵并进行 `α^T · client_matrix` 的矩阵乘法，避免了逐层循环聚合的开销；增量范数在 GPU 上通过展平向量减法 + 掩码一步完成。

---

### 5.2 Phase 2: Delta-Space 单边裁剪（去势）

**目标**：消除 Scaling Attack 的放大效应。与 v16 的全模型范数双边投影不同，v18f 在**增量空间**中执行**单边裁剪**——仅压缩超标更新，不放大弱更新。

#### 5.2.1 锚点选择

锚点定义为 Phase 1 搜索得到的最优权重 $\boldsymbol{\alpha}^*$ 中权重最大的客户端：
$$
a = \arg\max_{i \in \{1, \ldots, N\}} \alpha_i^*
$$

**设计动机**：$\alpha_a^*$ 最大意味着该客户端被 GA 搜索认为对全局模型性能贡献最大——即最"可信"的客户端。用它的增量范数作为裁剪上限。

#### 5.2.2 可训练参数掩码

裁剪仅作用于**可训练参数**（weights 与 biases，含 BN 的 γ 和 β），而 BN 的 `running_mean`、`running_var`、`num_batches_tracked` 等非可训练缓冲保持不变（直接 `layer.copy()`）。

定义可训练参数索引集 $\mathcal{T}$：由 GPU 加速器的 `trainable_mask` 给出（通过 `model.named_parameters()` 与 `model.state_dict()` 比较生成）。若不存在加速器，则所有浮点层均视为可裁剪。

函数 `_layer_is_scalable(layer, is_trainable)` 的精确逻辑：
- 若 `is_trainable is False`：返回 `False`（BN buffers 不参与）
- 否则：当 `layer.dtype` 为浮点或复数时返回 `True`

#### 5.2.3 锚点增量范数计算

$$
d_a = \|\boldsymbol{\Delta}_a\|_{\mathcal{T}} = \sqrt{\sum_{\ell \in \mathcal{T}} \|\mathbf{W}_{a,\ell} - \mathbf{W}_{t-1,\ell}\|_F^2}
$$

此值作为所有客户端增量幅度的上限。

#### 5.2.4 单边裁剪规则

对每个客户端 $i$，计算其增量范数：
$$
d_i = \|\boldsymbol{\Delta}_i\|_{\mathcal{T}} = \sqrt{\sum_{\ell \in \mathcal{T}} \|\mathbf{W}_{i,\ell} - \mathbf{W}_{t-1,\ell}\|_F^2}
$$

**裁剪判定**（单边）：
$$
\text{clipped}_i = \begin{cases} \text{True}, & d_i > d_a + \varepsilon_c \\ \text{False}, & \text{otherwise} \end{cases}
$$
其中 $\varepsilon_c = 10^{-12}$。

**缩放因子**（仅在裁剪时生效）：
$$
s_i = \begin{cases} \dfrac{d_a}{d_i + \varepsilon_p}, & \text{if clipped} \\ 1.0, & \text{otherwise} \end{cases}
$$

**层级裁剪（增量空间操作）**：
$$
\widetilde{\mathbf{W}}_{i,\ell} = \begin{cases}
\mathbf{W}_{t-1,\ell} + s_i \cdot (\mathbf{W}_{i,\ell} - \mathbf{W}_{t-1,\ell}), & \ell \in \mathcal{T},\; \text{clipped} \\
\mathbf{W}_{t-1,\ell} + (\mathbf{W}_{i,\ell} - \mathbf{W}_{t-1,\ell}), & \ell \in \mathcal{T},\; \text{not clipped} \\
\mathbf{W}_{i,\ell}, & \ell \notin \mathcal{T}
\end{cases}
$$

注意：未被裁剪的可训练层仍通过 $\mathbf{W}_{t-1} + \boldsymbol{\Delta}$ 的形式计算（结果等于原始 $\mathbf{W}_i$），这确保了数值路径的一致性。BN buffers（$\ell \notin \mathcal{T}$）直接保留客户端原值。

#### 5.2.5 v18f 单边裁剪 vs v16 双边投影的关键区别

| 维度 | v16（双边投影） | v18f（单边裁剪） |
|------|----------------|------------------|
| **操作空间** | 全模型范数空间 $\|\mathbf{W}_i\|$ | 增量空间 $\|\mathbf{W}_i - \mathbf{W}_{t-1}\|$ |
| **缩放方向** | 双边：$s_i$ 可 >1 或 <1 | 单边：$s_i \le 1$（永远不放大） |
| **基准范数** | 锚点全模型范数 $r_a = \|\mathbf{W}_a\|$ | 锚点增量范数 $d_a = \|\boldsymbol{\Delta}_a\|$ |
| **弱更新客户端** | 被放大（$s_i > 1$） → 可能放大噪声 | 保持原样 → 保留原始信噪比 |
| **公式** | $\widetilde{\mathbf{W}}_{i,\ell} = s_i \cdot \mathbf{W}_{i,\ell}$ | $\widetilde{\mathbf{W}}_{i,\ell} = \mathbf{W}_{t-1,\ell} + s_i \cdot \boldsymbol{\Delta}_{i,\ell}$ |

**为何改为单边**：v16 的双边投影将所有客户端的全模型范数统一到锚点范数 $r_a$。这对弱更新客户端是**放大**操作，可能放大其中的噪声分量。v18f 的单边裁剪仅压缩超标更新，保留了弱更新的原始信号，避免信噪比恶化。

**为何改为增量空间**：全模型范数衡量的是"参数有多大"，但在训练中后期，所有客户端的全模型范数都会比较接近（因为都从相似的全局模型出发训练），导致 v16 的投影退化为接近恒等映射，失去裁剪能力。增量范数衡量的是"这一轮改了多少"，更能区分正常更新与恶意放大。

#### 5.2.6 裁剪后不适用判定的回退

当 $\mathbf{W}_{t-1}$ 不可用（首轮）或锚点增量范数 $d_a$ 不可用（$d_a \le 0$ 或 NaN/Inf）时，所有客户端直接复制原始参数，不执行裁剪。

#### 5.2.7 裁剪后融合

在裁剪空间中，使用 Phase 1 的最优权重进行最终加权聚合：
$$
\mathbf{W}_{GA} = \sum_{i=1}^N \alpha_i^* \cdot \widetilde{\mathbf{W}}_i = \mathcal{A}(\{\widetilde{\mathbf{W}}_i\}, \boldsymbol{\alpha}^*)
$$

**Phase 1 与 Phase 2 的协同效应**：Phase 1 在原始参数空间中搜索权重，此时增量范数正则已经对 Scaling 攻击的大增量产生了适应度惩罚，使恶意客户端获得较低的 $\alpha_i^*$。Phase 2 进一步在增量空间中裁剪残留的超标更新。两个阶段从不同维度（适应度搜索 + 几何约束）协同防御 Scaling 类攻击。

---

### 5.3 Phase 3: BN 感知的全局动量平滑（平滑）

**目标**：通过服务端动量更新在时序维度上平滑训练轨迹，增强模型对单轮扰动和后期震荡的抵抗能力。同时正确处理 BatchNorm 层的缓冲区，避免 v16 中动量污染 BN running stats 的 bug。

VeriFL-v18f 采用 **FedOpt 范式** 的动量更新，维护两个跨轮状态缓冲区：
- $\mathbf{W}_{t-1}$：上一轮全局模型参数（`global_model_buffer`）
- $\mathbf{V}_{t-1}$：上一轮速度向量（`velocity_buffer`）

#### 5.3.1 首轮初始化

当缓冲区为空时（首轮聚合）：
$$
\mathbf{W}_t \leftarrow \mathbf{W}_{GA}, \qquad \mathbf{V}_t \leftarrow \mathbf{0}
$$
首轮不执行动量更新，直接采用 GA 聚合结果。

#### 5.3.2 后续轮次：BN 感知的分层更新

对每一层 $\ell$：

**情况 A：可训练参数（$\ell \in \mathcal{T}$，包括 Conv/Linear 权重和 BN γ, β）**

$$
\boldsymbol{\Delta}_{t,\ell} = \mathbf{W}_{GA,\ell} - \mathbf{W}_{t-1,\ell}
$$
$$
\mathbf{V}_{t,\ell} = \beta \cdot \mathbf{V}_{t-1,\ell} + \boldsymbol{\Delta}_{t,\ell}
$$
$$
\mathbf{W}_{t,\ell} = \mathbf{W}_{t-1,\ell} + \eta \cdot \mathbf{V}_{t,\ell}
$$

**情况 B：BN 缓冲区（$\ell \notin \mathcal{T}$，即 running_mean, running_var, num_batches_tracked）**

$$
\mathbf{V}_{t,\ell} = \mathbf{0}
$$
$$
\mathbf{W}_{t,\ell} = \mathbf{W}_{GA,\ell}
$$

即：BN 缓冲区**不施加动量**，直接使用 GA 聚合的原始值。速度缓冲区在该维度上始终为零。

**设计理由**：在 v16 中，动量被错误地应用于 BN 的 `running_mean` 和 `running_var`，导致这些统计量被"历史加速方向"污染，在长期训练（200 轮+）中出现推理性能退化。v18a 修复了此 bug，v18b/v18f 沿用。

#### 5.3.3 物理直觉

将全局模型参数类比为一个粒子在参数空间中运动：
- $\boldsymbol{\Delta}_t$ 是当前轮次的"推力"
- $\mathbf{V}_t$ 是累积的"速度"，保留了历史方向的惯性
- $\beta = 0.9$ 意味着速度在每轮衰减 10%，但保留 90% 的历史方向
- $\eta = 0.3$ 是"步长"，控制粒子实际移动的距离

这意味着：
- 如果连续多轮的 $\boldsymbol{\Delta}_t$ 方向一致（正常收敛），速度会在该方向上累积加速
- 如果某一轮的 $\boldsymbol{\Delta}_t$ 方向突变（如被攻击干扰），历史惯性会抵消部分偏移
- $\eta = 0.3 < 1$ 意味着全局模型不会完全跟随当前轮次的 GA 结果，而是保守地移动

#### 5.3.4 与 FedAvgM 的关系

VeriFL-v18f 的 Phase 3 本质上是 **FedAvgM（FedAvg with Server Momentum）** 的一个 BN 感知实例化。"伪梯度"方向为 $\mathbf{W}_{GA} - \mathbf{W}_{t-1}$，动量系数 $\beta = 0.9$，步长 $\eta = 0.3$。

---

### 5.4 后处理: BatchNorm 重校准

在 Phase 3 动量更新完成后，若模型包含 BatchNorm 层（如 ResNet20），则执行 BN 统计量重校准：

$$
\mathbf{W}_t \leftarrow \text{BNRecalibrate}(\mathbf{W}_t)
$$

**原因**：Phase 3 对可训练参数施加了动量，改变了模型权重，但 BN 的 `running_mean`/`running_var` 仍是 GA 聚合的原始值（各客户端非 IID 数据分布的加权混合）。重校准使 BN 统计量与新的权重参数匹配。

**实现细节**：
1. 调用 `GPUAccelerator.recalibrate_batchnorm(final_params)`
2. 将全部参数加载到 GPU 模型：`_load_state_from_ndarrays(params)`
3. 设置 `model.train()` 模式（BN 进入训练模式，forward 时会更新 running stats）
4. 在 `torch.no_grad()` 下遍历全部 500 个验证样本（`batch_size=64`，约 8 个 batch）
5. 恢复 `model.eval()` 模式
6. 导出更新后的全部参数（含刷新的 BN 统计量）

---

## 6. 完整伪代码

```
算法: VeriFL-v18f 三阶段防御聚合（Relative Delta-Norm 变体）

输入:
  - {W_i}_{i=1}^N: 本轮 N 个客户端上传的模型参数（含完整 state_dict）
  - W_{t-1}: 上一轮全局模型（首轮使用 initial_parameters，或 None）
  - V_{t-1}: 上一轮速度缓冲（首轮为 None）
  - fitness_fn: 服务端验证集评估函数 (params → (loss, accuracy))
  - trainable_mask: 可训练参数布尔掩码
  - λ_f=32.0, β=0.9, η=0.3, P=15, G=10: 超参数

输出:
  - W_t: 新的全局模型参数

────────────────────────────────────────

Phase 1: GA 零阶搜索 + Relative Delta-Norm

  // 获取全局参考模型
  W_ref ← W_{t-1}（若首轮则取 initial_parameters，若均无则 None）

  // 种群初始化
  P[0] ← [1/N, ..., 1/N]                    // FedAvg 基准
  P[1..4] ← sparse_probes(N, k∈{3,4,5})     // 稀疏探针
  P[5..P-1] ← random_normalized(N)           // 随机补齐

  α* ← None; F* ← -∞

  // GPU 预加载
  若有 GPU 加速器:
    将 {W_i} 展平为矩阵                        // set_client_parameters
    将 W_ref 展平并预算 ||W_ref||_T             // set_global_reference

  // 进化循环
  for g = 1 to G:
    for j = 1 to P:
      // 候选聚合
      W(α_j) ← Σ_i α_{j,i} · W_i           // GPU 矩阵乘法聚合
      L ← forward(W(α_j), val_set).loss      // GPU 前向推理

      若 L 为 NaN/Inf: F_j ← 0; continue

      // 增量范数计算（仅可训练参数）
      若 W_ref 可用:
        delta_norm ← ||W(α_j) - W_ref||_T
        global_prev_norm ← ||W_ref||_T
        relative_norm ← delta_norm / (global_prev_norm + ε_r)
      否则:
        relative_norm ← ||W(α_j)||_T / (global_prev_norm + ε_r)  // 回退

      C ← L + λ_f · relative_norm
      F_j ← 1 / (C + ε_f)

      若 F_j > F*:
        F* ← F_j; α* ← α_j

    // 停滞检查
    若 α* 为空 或 F* ≤ 0:
      重新初始化种群; continue

    // 遗传操作
    new_pop ← [α*]                            // 精英保留
    parents ← tournament_select(population, scores, k=2)
    while |new_pop| < P:
      (p1, p2) ← 从 parents 中取两个
      child ← linear_crossover(p1, p2)
      child ← gaussian_mutation(child, σ=0.05, p=0.1)
      child ← |child| / Σ child               // 归一化
      new_pop.append(child)
    population ← new_pop

  // 全局回退
  若 α* 为空: α* ← [1/N, ..., 1/N]

────────────────────────────────────────

Phase 2: Delta-Space 单边裁剪

  // 锚点选择
  a ← argmax_i α*_i

  // 锚点增量范数
  d_a ← ||W_a - W_ref||_T

  // 单边裁剪
  for i = 1 to N:
    d_i ← ||W_i - W_ref||_T

    若 W_ref 为空 或 d_a 无效:
      W̃_i ← copy(W_i); continue

    若 d_i > d_a + ε_c:
      clipped ← True
      s_i ← d_a / (d_i + ε_p)
    否则:
      clipped ← False
      s_i ← 1.0

    for ℓ = 1 to L:
      若 ℓ ∈ T:
        δ_ℓ ← W_{i,ℓ} - W_{ref,ℓ}           // 增量
        若 clipped:
          W̃_{i,ℓ} ← W_{ref,ℓ} + s_i · δ_ℓ   // 压缩增量
        否则:
          W̃_{i,ℓ} ← W_{ref,ℓ} + δ_ℓ          // 保持原样
      否则:
        W̃_{i,ℓ} ← copy(W_{i,ℓ})              // BN buffers 原样

Phase 2→3 融合:
  W_GA ← Σ_i α*_i · W̃_i

────────────────────────────────────────

Phase 3: BN 感知的全局动量平滑

  若 W_{t-1} 为 None:                         // 首轮
    W_t ← W_GA
    V_t ← 0
  否则:                                        // 后续轮次
    for ℓ = 1 to L:
      若 trainable_mask[ℓ] = False:            // BN buffers
        V_{t,ℓ} ← 0
        W_{t,ℓ} ← W_{GA,ℓ}                    // 直接用聚合值
      否则:                                     // 可训练参数
        Δ_{t,ℓ} ← W_{GA,ℓ} - W_{t-1,ℓ}
        V_{t,ℓ} ← β · V_{t-1,ℓ} + Δ_{t,ℓ}
        W_{t,ℓ} ← W_{t-1,ℓ} + η · V_{t,ℓ}

  // 更新跨轮缓冲
  global_model_buffer ← clone(W_t)
  velocity_buffer ← V_t

后处理:
  若模型含 BatchNorm:
    W_t ← BN_recalibrate(W_t)                // 在验证集上刷新 BN 统计量

返回 W_t
```

---

## 7. 超参数完整字典

### 7.1 可配置超参数

| 参数名 | 类型 | 默认值 | 作用 | 配置位置 |
|--------|------|--------|------|---------|
| `pop_size` | int | 15 | GA 每代候选权重向量数量，控制搜索覆盖率与计算成本 | YAML `params.pop_size` |
| `generations` | int | 10 | 每轮联邦聚合内的 GA 进化代数，控制搜索深度 | YAML `params.generations` |
| `lambda_reg` | float | **32.0** | 适应度函数中相对增量范数惩罚强度 $\lambda_f$ | YAML `params.lambda_reg` |
| `server_momentum` | float | 0.9 | FedOpt 速度项保留系数 $\beta$，控制跨轮惯性 | 代码级默认 |
| `server_lr` | float | 0.3 | 服务器动量更新步长 $\eta$，控制全局参数移动幅度 | 代码级默认 |
| `debug_probe_enabled` | bool | False | 是否启用诊断探针（输出详细中间数据） | YAML `params.debug_probe_enabled` |

### 7.2 实现固定常量

| 参数 | 值 | 作用 |
|------|---|------|
| `mutation_prob` | 0.1 | 子代触发高斯扰动的概率 |
| `mutation_sigma` | 0.05 | 高斯变异噪声标准差 |
| `tournament_k` | 2 | 锦标赛大小（选择压力） |
| `ε_f` (eps_fitness) | $10^{-12}$ | 适应度分母稳定项 |
| `ε_r` (eps_relative) | $10^{-8}$ | 相对范数分母稳定项 |
| `ε_p` (eps_projection) | $10^{-9}$ | Phase 2 缩放分母稳定项 |
| `ε_c` (eps_clip) | $10^{-12}$ | Phase 2 裁剪判定阈值 |
| `ε_n` (eps_normalize) | $10^{-9}$ | 归一化分母稳定项 |
| `bn_batch_size` | 64 | BN 重校准的批量大小 |
| `bn_passes` | 1 | BN 重校准的前向传播遍历次数 |

### 7.3 实验配置参数（非算法层面）

| 参数 | 典型值 | 作用 |
|------|--------|------|
| `num_clients` | 10 | 联邦学习参与客户端总数 |
| `num_malicious` | 3 | 恶意客户端数量（30%） |
| `rounds` | 200 | 联邦训练总轮数 |
| `alpha` (Dirichlet) | 0.5 | Non-IID 异质性参数 |
| `server_val_ratio` | 500 | 服务端验证集样本数 |
| `server_trust_ratio` | 500 | 服务端可信数据集样本数 |
| `client_lr` | 0.01 | 客户端本地 SGD 学习率 |
| `client_momentum` | 0.9 | 客户端本地 SGD 动量 |

---

## 8. 适应度函数详解

### 8.1 适应度计算流程

适应度计算通过 GPU 加速器完成：客户端参数在每轮开始时展平为 GPU 矩阵，全局参考模型 $\mathbf{W}_{t-1}$ 同步加载到 GPU。每次评估仅需一次矩阵乘法（聚合）+ 一次前向推理（损失）+ 一次向量减法 + 掩码运算（增量范数），全程无 CPU-GPU 数据搬运。

适应度计算的完整流程：

```
1. GPU 矩阵乘法聚合:
   aggregated_flat = α^T · client_params_matrix   // GPU matmul

2. 加载聚合参数到模型，执行 GPU 前向推理:
   val_loss = CrossEntropyLoss(model(val_images), val_labels)

3. 计算增量范数 (GPU 向量运算, 仅可训练参数):
   delta_flat = aggregated_flat - global_prev_flat
   delta_norm = √(Σ delta_flat[T]²)

4. 获取上一轮全局模型范数 (每轮预算一次，跨候选解共享):
   global_prev_norm = √(Σ global_prev_flat[T]²)

5. 计算相对增量范数:
   relative_norm = delta_norm / (global_prev_norm + 1e-8)

6. 计算代价与适应度:
   cost = val_loss + λ_f · relative_norm
   fitness = 1 / (cost + 1e-12)
```

### 8.2 回退逻辑

| 情况 | 回退行为 |
|------|---------|
| `val_loss` 为 NaN/Inf | fitness = 0（淘汰此候选解） |
| `delta_norm` 为 NaN/Inf | 退化为 `model_norm`（全模型范数）替代增量范数 |
| `W_ref` 不可用（首轮） | 使用全模型范数作为 `regularizer_norm` |
| `global_prev_norm` = 0 | 直接使用 `regularizer_norm` 不做除法 |

### 8.3 相对于 v16 的适应度扩展

v18f 在 v16 适应度基础上新增了增量范数分支，GPU 加速器相应扩展：

| 能力 | v16 | v18f |
|------|-----|------|
| 矩阵乘法聚合 | ✅ `α^T · client_matrix` | ✅ 相同 |
| GPU 前向推理 | ✅ | ✅ 相同 |
| 全模型范数 | ✅ `model.parameters()` | ✅ `trainable_mask` 过滤 |
| 增量范数 | — | ✅ `aggregated_flat - global_prev_flat` |
| 全局参考模型 | — | ✅ `set_global_reference()` 每轮加载一次 |

---

## 9. GPU 加速架构

### 9.1 GPUAccelerator 在 v18f 中的角色

GPU 加速器是 VeriFL 系列的核心计算引擎，负责所有密集运算。v18f 在 v16 基础上扩展了增量范数计算能力：

| 能力 | 方法 | 说明 |
|------|------|------|
| 客户端参数矩阵化 | `set_client_parameters()` | 展平为 `(N, D)` GPU 矩阵 |
| 全局参考模型加载 | `set_global_reference()` | 展平 $\mathbf{W}_{t-1}$ 并预算 $\|\mathbf{W}_{t-1}\|_{\mathcal{T}}$ |
| 快速适应度计算 | `calculate_fitness(alpha)` | 矩阵乘法聚合 + 前向推理 + 增量范数，返回 `(loss, model_norm, delta_norm)` |
| BN 统计量校准 | `recalibrate_batchnorm()` | 在验证集上刷新 running stats |
| 可训练参数识别 | `trainable_mask` / `flat_trainable_mask` | 布尔掩码，用于范数计算 |
| BN 检测 | `has_batchnorm` | 决定是否执行 BN 重校准 |

### 9.2 初始化流程

```
StrategyBuilder.build()
  └→ _build_ga_strategy()
       └→ strategy.initialize_gpu_accelerator(model_template, val_data)
            └→ GPUAccelerator.__init__(model_template, val_data)
                 ├─ 深拷贝模型到 GPU
                 ├─ 预加载验证集到 GPU (val_images, val_labels)
                 ├─ 计算 trainable_mask (逐层) + flat_trainable_mask (展平)
                 │   (named_parameters ∩ state_dict 的差集)
                 ├─ 记录参数形状和大小
                 └─ 检测是否含 BatchNorm
```

---

## 10. 数据流全景

### 10.1 数据分割架构

```
原始 CIFAR-10 训练集 (50,000 样本)
    │
    ├── 服务端验证集 (500 样本, 固定种子洗牌选取)
    │     ├─ 用途1: GA 适应度评估 (fitness_fn = Evaluator.evaluate)
    │     ├─ 用途2: GPU 加速器预加载 (val_images, val_labels)
    │     ├─ 用途3: BN 统计量重校准
    │     └─ 数据增强: RandomCrop(32,padding=4) + RandomHorizontalFlip + Normalize
    │
    ├── 服务端可信数据集 (500 样本, 与验证集互斥)
    │     └─ 用途: FLTrust 可信更新 (v18f 不使用)
    │
    └── 客户端训练池 (≈49,000 样本, 与以上两者互斥)
          └─ Dirichlet(α=0.5) Non-IID 分配给 N 个客户端
               ├─ 客户端 0..K-1 (恶意): 正常数据 + 攻击逻辑
               └─ 客户端 K..N-1 (良性): 正常 SGD 训练

原始 CIFAR-10 测试集 (10,000 样本, 完全封存)
    ├─ 用途1: 每轮泛化评估 (evaluate_test)
    ├─ 用途2: 后门 ASR 计算 (evaluate_backdoor_test)
    └─ 数据增强: 仅 ToTensor + Normalize (无随机操作)
```

**数据互斥断言**：系统通过显式断言确保 `server_val ∩ client_pool = ∅`、`server_trust ∩ client_pool = ∅`、`server_val ∩ server_trust = ∅`。

### 10.2 每轮数据流

```
         ┌──────────────────────────────────────────────┐
Round t   │                   服务器                      │
         │                                              │
         │   收到: results = [(proxy_1, FitRes_1), ...]  │
         │                                              │
         │   ┌───── aggregate_fit() ─────────────────┐  │
         │   │                                       │  │
         │   │  Step A: 提取 {W_i}                   │  │
         │   │   W_i = parameters_to_ndarrays(       │  │
         │   │          FitRes_i.parameters)          │  │
         │   │                                       │  │
         │   │  Step A': GPU 预加载                   │  │
         │   │   gpu.set_client_parameters({W_i})    │  │
         │   │   gpu.set_global_reference(W_{t-1})   │  │
         │   │                                       │  │
         │   │  Phase 1: GA 搜索 → α*                │  │
         │   │   对每个候选 α:                         │  │
         │   │     (L, norm, Δnorm) =                │  │
         │   │       gpu.calculate_fitness(α)         │  │
         │   │     delta = ||W(α) - W_ref||_T         │  │
         │   │     prev = ||W_ref||_T                 │  │
         │   │     cost = L + λ_f · delta/(prev+ε)    │  │
         │   │     F = 1 / (cost + ε)                 │  │
         │   │                                       │  │
         │   │  Phase 2: Delta-Space 单边裁剪         │  │
         │   │   a = argmax(α*)                       │  │
         │   │   d_a = ||W_a - W_ref||_T             │  │
         │   │   for each i:                          │  │
         │   │     d_i = ||W_i - W_ref||_T           │  │
         │   │     若 d_i > d_a:                      │  │
         │   │       W̃_i = W_ref + (d_a/d_i)·Δ_i    │  │
         │   │     否则:                              │  │
         │   │       W̃_i = W_i                       │  │
         │   │                                       │  │
         │   │  融合: W_GA = Σ α*_i · W̃_i            │  │
         │   │                                       │  │
         │   │  Phase 3: BN 感知动量                  │  │
         │   │   可训练层:                             │  │
         │   │     V_t = β·V_{t-1} + (W_GA-W_{t-1})  │  │
         │   │     W_t = W_{t-1} + η·V_t             │  │
         │   │   BN buffers:                          │  │
         │   │     W_t = W_GA                         │  │
         │   │                                       │  │
         │   │  BN 校准: recalibrate(W_t)            │  │
         │   │                                       │  │
         │   └───────────────────→ Parameters ────────┤  │
         │                                              │
         │   evaluate_fn(W_t) → loss, acc, asr         │
         │                                              │
         │   下发 W_t 给所有客户端                       │
         └──────────────────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         Client 0     Client 1     ...  Client N-1
         (恶意)       (恶意)            (良性)
              │            │            │
              └────────────┼────────────┘
                           ▼
                     Round t+1 ...
```

---

## 11. 威胁模型与防御目标

### 11.1 威胁模型

VeriFL-v18f 假设以下威胁场景：

- **攻击者能力**：
  - 完全控制 $K$ 个客户端（$K < N/2$），可以任意修改其上传的模型参数
  - 知晓联邦学习的通信协议和聚合方式（白盒）
  - 可以选择在任意轮次发起攻击

- **攻击者限制**：
  - 无法访问服务端验证集
  - 无法修改其他良性客户端的训练过程或上传参数
  - 无法干扰服务端聚合逻辑

- **服务器假设**：
  - 服务器是诚实的（honest-but-cautious）
  - 服务器拥有一个小型干净验证集（500 样本）
  - 服务器可以在每轮聚合时执行额外的计算（GA 搜索 + 裁剪 + 动量）

### 11.2 防御目标

1. **主任务准确率维护**（Main Task Accuracy, MTA）：在拜占庭攻击下维持全局模型在未污染测试集上的分类准确率
2. **后门攻击抑制**（Attack Success Rate, ASR）：降低后门攻击的成功率
3. **收敛稳定性**：保持训练过程的平稳收敛，避免剧烈震荡

---

## 12. 对各类攻击的防御机制分析

### 12.1 Byzantine Attack（随机噪声/对抗性梯度）

| 阶段 | 防御机制 |
|------|---------|
| Phase 1 | 随机噪声方向的更新在验证集上导致高损失 + 大增量 → 双重惩罚 → 极低适应度 → 极低权重 $\alpha_i$ |
| Phase 2 | 大增量超过锚点增量范数 → 被裁剪到安全半径内 |
| Phase 3 | 动量抵消单轮噪声冲击 |

### 12.2 Label Flip Attack（标签翻转）

| 阶段 | 防御机制 |
|------|---------|
| Phase 1 | 翻转标签训练的更新偏离正确梯度方向，聚合后验证损失上升 → GA 自动降权 |
| Phase 2 | 增量范数通常无异常（标签翻转不改变更新幅度），裁剪对此攻击效果有限，主要依赖 Phase 1 |
| Phase 3 | 动量平滑减缓标签翻转对全局模型方向的偏移速度 |

### 12.3 Scaling Attack（梯度放大）

| 阶段 | 防御机制 |
|------|---------|
| Phase 1 | $\lambda_f = 32.0$ 的相对增量范数惩罚使放大后的增量在适应度中受到显著惩罚（增量范数正比于放大倍数） |
| Phase 2 | **核心防线**。增量范数 $d_i \gg d_a$ → 被裁剪到 $d_a$ 范围内，$s_i \ll 1$ |
| Phase 3 | 即使残余影响通过，动量的保守步长 $\eta = 0.3$ 限制了单轮偏移 |

### 12.4 Backdoor Attack / Model Replacement

| 阶段 | 防御机制 |
|------|---------|
| Phase 1 | 后门更新在干净验证集上的损失可能升高 → 降权。Model Replacement 的大尺度更新触发增量范数惩罚 |
| Phase 2 | Model Replacement 通常伴随显著的增量放大 → 被单边裁剪压缩 |
| Phase 3 | 后门效果需要持续多轮注入；动量惯性使得全局模型不会被单轮后门彻底修改 |

### 12.5 防御协同效应

```
Phase 1（搜索）→ 提供两个关键输出:
  ├─ α*: 最优权重分配（降权恶意客户端）    → 融合使用
  └─ a = argmax(α*): 最可信客户端索引     → Phase 2 使用

Phase 2（裁剪）→ 在增量空间约束更新幅度:
  └─ {W̃_i}: 裁剪后的参数                 → 融合使用

融合 → 结合 Phase 1 + Phase 2 的结果:
  └─ W_GA: 去恶意化的聚合参数             → Phase 3 使用

Phase 3（平滑）→ 在时序维度上平滑 + BN 校准:
  └─ W_t: 最终全局模型
```

单独看每个阶段都有其局限（如 Phase 1 可能被精心构造的攻击欺骗，Phase 2 只约束增量范数不约束方向），但三者组合覆盖了权重空间（Phase 1）、增量空间（Phase 2）、时间维度（Phase 3）三个正交维度。

---

## 13. 与经典方法的关键区别

### 13.1 与 FedAvg 的区别

| 维度 | FedAvg | VeriFL-v18f |
|------|--------|------------|
| 权重分配 | 按样本数等比例 | GA 搜索最优权重 |
| 异常防御 | 无 | 三阶段协同防御 |
| 更新约束 | 无 | 增量空间单边裁剪 |
| 跨轮状态 | 无 | 全局模型缓冲 + 速度缓冲 |
| BN 处理 | 简单平均 | BN 感知动量 + 重校准 |

### 13.2 与 Multi-Krum 的区别

| 维度 | Multi-Krum | VeriFL-v18f |
|------|-----------|------------|
| 过滤机制 | 基于欧式距离剔除异常值（0/1 决策） | GA 搜索连续权重（0 到 1 之间） |
| 参考标准 | 客户端之间的互相距离 | 服务端验证集的损失 + 增量范数 |
| 利用程度 | 被选中的客户端等权 | 每个客户端有精细的差异化权重 |
| 模长约束 | 无 | 增量空间单边裁剪 |

### 13.3 与 Trimmed Mean / Median 的区别

| 维度 | Trimmed Mean/Median | VeriFL-v18f |
|------|-------------------|------------|
| 操作粒度 | 逐坐标独立处理 | 模型整体（跨层）优化 |
| 对齐方式 | 统计位置估计 | 验证集驱动的优化搜索 |
| Scaling 防御 | 部分有效（中位数具有鲁棒性） | 双重防御（增量范数正则 + 单边裁剪） |

### 13.4 与 FLTrust 的区别

| 维度 | FLTrust | VeriFL-v18f |
|------|---------|------------|
| 信任来源 | 在可信数据上训练的"黄金更新" | 验证集上的损失评估 |
| 权重计算 | 余弦相似度归一化 | GA 搜索优化 |
| 模长约束 | 按可信更新范数归一化（双边） | 按锚点增量范数裁剪（单边） |
| 额外数据需求 | 需要可信训练数据 + 训练过程 | 仅需验证集 + 推理（不训练） |

### 13.5 与 Zeno 的区别

| 维度 | Zeno | VeriFL-v18f |
|------|------|------------|
| 优化方式 | 逐客户端独立评分后排序 | GA 在组合空间中全局搜索 |
| 损失评估 | 评估每个客户端更新对全局模型的贡献 | 评估权重组合的整体效果 |
| 模长处理 | Zeno++ 加入梯度范数惩罚 | 相对增量范数正则 + 单边裁剪 |
| 量纲适应 | 固定惩罚系数 | 自适应 $\lambda_{\text{eff}} = \lambda_f / \|\mathbf{W}_{t-1}\|$ |

---

## 14. 继承体系与代码架构

### 14.1 类图

```
flwr.server.strategy.FedAvg
│   ├─ configure_fit()      // Flower 接口
│   ├─ configure_evaluate()  // Flower 接口
│   └─ aggregate_fit()      // 被子类覆盖
│
└── MicroGABase (src/strategies/ours/ga_base.py)
    │   成员变量:
    │   ├─ fitness_fn: Callable    // 验证集评估函数
    │   ├─ pop_size: int           // 种群大小
    │   ├─ generations: int        // 进化代数
    │   ├─ lambda_reg: float       // 正则化系数
    │   ├─ loss_power: float       // 损失幂次（v18f 未使用）
    │   └─ gpu_accelerator: Optional[GPUAccelerator]
    │
    │   方法:
    │   ├─ initialize_gpu_accelerator()  // GPU 加速器初始化
    │   ├─ _init_population()            // 默认种群初始化
    │   ├─ _tournament_selection()       // 锦标赛选择（公共）
    │   ├─ _crossover()                  // 线性交叉（公共）
    │   ├─ _mutation()                   // 高斯变异（公共）
    │   ├─ _record_weight_telemetry()    // 权重遥测记录
    │   ├─ calculate_fitness()           // 抽象接口
    │   └─ aggregate_fit()              // 默认模板
    │
    └── StrategyV18FRelative (src/strategies/ours/v18f_relative.py)
        │   新增成员变量:
        │   ├─ server_momentum: float                    // 动量系数 β (0.9)
        │   ├─ server_lr: float                          // 服务端学习率 η (0.3)
        │   ├─ debug_probe_enabled: bool                 // 诊断开关
        │   ├─ probe_counterfactual_gammas: List[float]   // 反事实探针倍率
        │   ├─ global_model_buffer: List[ndarray]         // 全局模型缓冲
        │   ├─ velocity_buffer: List[ndarray]             // 速度缓冲
        │   ├─ probe_fitness_rows: List[Dict]             // 适应度探针数据
        │   ├─ probe_phase2_rows: List[Dict]              // Phase 2 探针数据
        │   ├─ probe_phase_mismatch_rows: List[Dict]      // 阶段失配探针
        │   └─ probe_counterfactual_rows: List[Dict]      // 反事实探针
        │
        │   覆盖方法:
        │   ├─ _init_population()          // 含稀疏探针的种群初始化
        │   ├─ calculate_fitness()         // 委托给 _compute_fitness_details
        │   └─ aggregate_fit()            // 完整三阶段防御逻辑
        │
        │   新增方法:
        │   ├─ _compute_fitness_details()  // 完整适应度计算（含相对增量范数）
        │   ├─ _calc_l2_norm()            // 可训练参数 L2 范数
        │   ├─ _calc_delta_l2_norm()      // 可训练参数增量 L2 范数
        │   ├─ _layer_is_scalable()       // 层是否可参与缩放/裁剪
        │   ├─ _clone_param_list()        // 深拷贝参数列表
        │   ├─ _get_global_reference_params() // 获取全局参考模型
        │   └─ get_probe_tables()         // 导出诊断表
        │
        └── 协作类:
            ├─ GPUAccelerator (src/strategies/ours/gpu_accelerator.py)
            │   ├─ set_client_parameters()
            │   ├─ recalibrate_batchnorm()
            │   ├─ trainable_mask
            │   └─ has_batchnorm
            │
            └─ aggregate_weighted() (src/strategies/utils.py)
                └─ 支持 torch.Tensor 和 np.ndarray 的加权聚合
```

### 14.2 策略注册与发现

```python
# src/strategies/__init__.py
STRATEGY_REGISTRY = {
    ...
    "v18f_relative": StrategyV18FRelative,
    ...
}
```

### 14.3 aggregate_weighted 的实现细节

`aggregate_weighted(weights_results, alpha)` 执行层级加权聚合：

```python
aggregated[ℓ] = Σ_i alpha_i · weights_results[i][ℓ]
```

特殊处理：
- 非浮点/非复数层（如 BN 的 `num_batches_tracked`，dtype 为 int64）：直接取权重最大客户端的值
- 同时支持 PyTorch Tensor 和 NumPy ndarray 两种输入格式

### 14.4 诊断探针系统

v18f 内置了可选的诊断探针（`debug_probe_enabled=True` 时启用），输出四类 CSV 数据：

| 探针表 | 内容 | 目的 |
|--------|------|------|
| `fitness_breakdown.csv` | 每代每个候选解的 loss、model_norm、delta_norm、relative_norm、cost、fitness | 分析 GA 搜索行为 |
| `phase2_norm_projection.csv` | 每个客户端的原始/裁剪后范数、缩放因子、是否裁剪 | 验证 Phase 2 裁剪效果 |
| `phase_mismatch.csv` | Phase 1→2 过渡中的 pre/post cost/fitness 对比 | 检测两阶段不一致性 |
| `counterfactual_scaling_probe.csv` | 模拟不同放大倍率下的裁剪行为 | 评估对缩放攻击的鲁棒性 |

---

## 15. 实验环境设定

### 15.1 数据集

| 属性 | 值 |
|------|----|
| 数据集 | CIFAR-10 |
| 训练集大小 | 50,000 |
| 测试集大小 | 10,000 |
| 类别数 | 10 |
| 输入尺寸 | 3 × 32 × 32 |
| 归一化 | Mean=(0.4914, 0.4822, 0.4465), Std=(0.2023, 0.1994, 0.2010) |

### 15.2 模型

| 模型 | 结构 | 适用场景 |
|------|------|---------|
| **ResNet20** | 3 层残差网络 (16→32→64 通道)，含 BatchNorm，约 0.27M 参数 | 正式实验 |

ResNet20 结构：
```
Input (3×32×32)
  → Conv1 (3→16, 3×3, BN, ReLU)
  → Layer1: 3 × BasicBlock (16→16, BN)
  → Layer2: 3 × BasicBlock (16→32, stride=2, BN)
  → Layer3: 3 × BasicBlock (32→64, stride=2, BN)
  → AvgPool(8×8)
  → FC (64→10)
```

其中含 21 个 BatchNorm2d 层，可训练参数（Conv weights + BN γ/β + Linear weight/bias），非可训练缓冲（BN running_mean/running_var/num_batches_tracked）。

### 15.3 数据分配

| 属性 | 值 |
|------|----|
| 客户端总数 $N$ | 10 |
| 分配方式 | Dirichlet Non-IID ($\alpha = 0.5$) |
| 服务端验证集 | 从训练集中抽取 500 样本 |
| 服务端可信数据 | 从训练集中抽取 500 样本（与验证集互斥） |
| 客户端训练池 | ≈49,000 样本 |

### 15.4 客户端训练配置

| 属性 | 值 |
|------|----|
| 优化器 | SGD |
| 学习率 | 0.01 |
| 动量 | 0.9 |
| 损失函数 | CrossEntropyLoss |
| 每轮训练 | 1 个 epoch（遍历本地数据集一次） |

### 15.5 攻击场景

| 场景 | 攻击类型 | 恶意客户端数 | 其他参数 |
|------|---------|------------|---------|
| Byzantine | 随机/对抗性梯度 | 3/10 | — |
| Label Flip | 标签翻转 ($y \to 9-y$) | 3/10 | — |
| Backdoor | 像素触发器 + 目标标签 | 4/10 | `poison_ratio=0.5`, `trigger_size=3`, `eps=2.0` |
| Scaling | 梯度倍数放大 | 3-4/10 | 倍率由攻击实现控制 |
| Model Replacement | 放大后门更新以覆盖 FedAvg | 1/10 | — |

### 15.6 评估指标

| 指标 | 计算方式 | 数据源 |
|------|---------|--------|
| **Loss** | CrossEntropyLoss | 封存测试集 |
| **Accuracy** | 正确分类数 / 总样本数 | 封存测试集 |
| **ASR** (Attack Success Rate) | 将触发器注入非目标样本后预测为目标标签的比例 | 封存测试集 |

### 15.7 实验结果输出

```
results/{ENV_NAME}/{SCENARIO_NAME}/{STRATEGY_DIR}/seed_{SEED}/
├── metrics.csv    # 每轮指标: round, loss, accuracy, asr, time
└── config.yaml    # 实验配置快照
```

---

## 16. 隐含假设与已知局限

### 16.1 锚点可信性假设

Phase 2 完全依赖 $a = \arg\max_i \alpha_i^*$。若 Phase 1 的 GA 搜索被以下情况误导：
- 恶意客户端精心构造了低损失但含后门的更新
- 验证集太小或偏斜导致评估不准确

则错误的锚点将提供错误的裁剪上限。但相比 v16 的双边投影（错误锚点会把所有人缩放到错误半径），v18f 的单边裁剪仅影响超标客户端，不会放大小更新客户端，错误传播范围更小。

### 16.2 "增量范数异常 ≈ 攻击"假设

单边裁剪仅约束增量的范数，不约束方向。攻击者如果能在保持正常增量范数的同时注入方向性偏差（如定向对齐攻击），则可以穿透 Phase 2。此时防御主要依赖 Phase 1 的适应度评估。

### 16.3 全零适应度停滞风险

极端高损失或 NaN/Inf 可导致整个种群的适应度全部为 0。实现仅做"重置种群重新搜索"，缺少自适应退火或温度调控机制。

### 16.4 首轮回退行为

首轮缺少 `global_model_buffer`，增量范数不可用。此时：
- Phase 1 回退到全模型范数除以初始参数范数作为正则项
- Phase 2 因无全局参考模型，直接跳过裁剪  

首轮因此仅有 Phase 1 的适应度搜索和 Phase 3 的初始化（无动量），防御能力较弱。

### 16.5 动量累积污染风险

速度缓冲 $\mathbf{V}_t$ 跨轮记忆历史方向。若攻击者持续多轮注入一致的定向偏移，$\mathbf{V}_t$ 可能在该方向上持续累积，形成"惯性陷阱"。$\beta = 0.9$ 的衰减意味着需要约 10 轮才能使历史速度衰减到 $< 35\%$。

### 16.6 failures 参数未利用

Flower 框架传入的 `failures` 列表（客户端训练失败信息）未进入任何决策逻辑。

### 16.7 相对正则在模型范数极小时的行为

当全局模型范数 $\|\mathbf{W}_{t-1}\|_{\mathcal{T}} \to 0$ 时（理论上不会发生，但极端初始化可能触发），$\lambda_{\text{eff}} \to \infty$，正则项会主导代价函数，使 GA 搜索过度保守。$\varepsilon_r = 10^{-8}$ 的分母稳定项提供了有限的保护。

---

## 17. 从 v16 到 v18f 的演化脉络

### 17.1 主线版本

| 版本 | 核心特性 | 问题/教训 | 下一版如何改进 |
|------|---------|---------|--------------|
| **v16 (Three-Phase)** | 三阶段防御：GA + 全模型双边投影 + 全参数动量 | BN 动量污染；双边投影在训练后期退化；全模型范数对尺度敏感 | → v18a |
| **v18a (Phase2-Delta)** | 修复 BN 动量 bug；Phase 2 改为增量空间裁剪 | Phase 1 仍用绝对模型范数 | → v18b |
| **v18b (Aligned)** | Phase 1 改为增量范数正则 $\lambda=1.0$ | λ 为固定值，对不同模型/训练阶段不自适应 | → v18f |
| **v18c (Scale-Free)** | 尝试去除范数正则依赖 | 防御失效，无正则导致 GA 无法区分恶意 | → 废弃 |
| **v18d (Rank Fusion)** | 尝试秩融合替代加权平均 | 200R 下不收敛 | → 废弃 |
| **v18f (Relative)** | 相对增量范数 $\lambda_f=32.0$；量纲自适应 | **当前活跃版本** | — |

### 17.2 v18f 的核心设计决策总结

1. **Relative Delta-Norm**：$\text{cost} = L_{val} + \lambda_f \cdot \frac{\|\boldsymbol{\Delta}\|_{\mathcal{T}}}{\|\mathbf{W}_{t-1}\|_{\mathcal{T}} + \varepsilon_r}$，使正则力度对模型尺度具有不变性，$\lambda_f = 32.0$ 在 ResNet-20 上等价于 $\lambda_{\text{eff}} \approx 1.0$。

2. **Delta-Space 单边裁剪**：仅裁剪增量超标的客户端（`d_i > d_a`），不放大弱更新。在增量空间中操作：$\widetilde{\mathbf{W}} = \mathbf{W}_{t-1} + s \cdot (\mathbf{W} - \mathbf{W}_{t-1})$。

3. **BN 感知动量**：可训练参数施加 FedOpt 动量；BN buffers 跳过动量，直接用聚合值 + recalibration 覆盖。

4. **GPU 加速适应度**：GPU 加速器负责矩阵乘法聚合 + 前向推理 + 增量范数计算，全部操作在 GPU 上完成，无 CPU-GPU 数据搬运。

---

> **文档结束**  
> 本文档基于 ShieldFL 项目代码库（`src/strategies/ours/v18f_relative.py` 及其所有依赖模块）严格生成。所有公式、伪代码、参数表均与代码实现一一对应。与 v16 algorithm.md 的差异经逐行代码核验确认。
