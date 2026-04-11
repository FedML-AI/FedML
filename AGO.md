# ShieldFL 核心算法复现文档

## VeriFL-v16: Validation-Driven Anchor-Momentum Strategy

### 三阶段拜占庭鲁棒联邦学习聚合策略

---

## 1. 算法概述

VeriFL-v16 是一种基于遗传算法（Micro-GA）的拜占庭鲁棒联邦学习聚合策略。核心思想是：**在服务端利用微型遗传算法搜索最优客户端权重组合，隐式排除恶意客户端，并通过锚点投影和全局动量平滑增强鲁棒性。**

算法采用三阶段防御流水线：

| 阶段 | 名称 | 功能 |
|------|------|------|
| **Phase 1** | 侦查（GA 零阶搜索） | 利用遗传算法在验证集上搜索最优客户端权重，通过 Loss + L2 正则化的适应度函数，迫使恶意大梯度客户端获得极低权重 |
| **Phase 2** | 去势（锚点归一化投影） | 锁定 GA 最信任的客户端为锚点，将所有客户端梯度投影到锚点的超球面上，抵御 Scaling Attack |
| **Phase 3** | 平滑（FedOpt 服务器动量） | 对全局模型参数应用指数移动平均（EMA/Momentum），消除后期训练震荡 |

---

## 2. 系统架构

### 2.1 端到端执行流

```
CLI 入口 (main.py)
  │
  ├── 配置加载: scenario.yaml + method.yaml → Config 数据类
  ├── 攻击管理器: AttackManager(type, num_malicious, params)
  └── 仿真运行器: SimulationRunner(config, attack_manager)
        │
        └── for seed in seeds:
              for strategy in strategies:
                1. 设定随机种子
                2. DataFactory.create() → DataBundle
                   (train/val/trust/test 互斥分割)
                3. ModelFactory → 初始模型权重
                4. Evaluator(model_template, val_loader, test_loader, ...)
                5. StrategyBuilder.build("v16", ...) → StrategyV16
                   └── 初始化 GPUAccelerator(model_template, val_data)
                6. client_fn → BenignClient / MaliciousClient
                7. fl.simulation.start_simulation()
                   ├── 每轮: 客户端本地 SGD 训练
                   ├── 恶意客户端: AttackManager.execute() → 毒化权重
                   ├── 服务端: StrategyV16.aggregate_fit()
                   │   Phase 1: GA 搜索 → best_weights
                   │   Phase 2: 锚点投影 → projected_weights
                   │   Phase 3: EMA 动量 → final_params
                   │   (BN 校准 → recalibrate_batchnorm)
                   └── evaluate_fn: 记录 loss/acc/ASR
                8. MetricMonitor.save() → metrics.csv
```

### 2.2 核心模块依赖关系

```
StrategyV16 (v16.py)
  │
  ├── 继承 MicroGABase (ga_base.py) ← 继承 flwr.server.strategy.FedAvg
  │     ├── _tournament_selection()   — 锦标赛选择
  │     ├── _crossover()              — 线性交叉
  │     └── _mutation()               — 高斯变异
  │
  ├── 覆写 _init_population()        — FedAvg + One-Hot 探针 + 随机
  ├── 覆写 calculate_fitness()        — 1/(Loss + λ·‖w‖₂ + ε)
  ├── 覆写 aggregate_fit()           — 三阶段完整流水线
  │
  ├── 使用 GPUAccelerator (gpu_accelerator.py)
  │     ├── set_client_parameters()   — GPU 矩阵预加载
  │     ├── calculate_fitness()       — GPU 矩阵乘法加速适应度计算
  │     ├── recalibrate_batchnorm()   — BN running stats 校准
  │     └── trainable_mask            — 可训练参数掩码（区分 BN buffers）
  │
  ├── 使用 aggregate_weighted() (strategies/utils.py)
  │     └── 加权聚合客户端参数
  │
  └── 接收 fitness_fn (来自 Evaluator.evaluate)
        └── evaluate(params) → (loss, accuracy)
```

---

## 3. 数据准备

### 3.1 数据集加载与分割

使用 CIFAR-10 数据集，训练集共 50,000 张图片，测试集 10,000 张。

**预处理管道：**
- **训练集客户端数据**：ToTensor → Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
- **服务器验证集**：RandomCrop(32, padding=4) → RandomHorizontalFlip → ToTensor → Normalize
- **测试集**：ToTensor → Normalize（严禁随机增强）

**互斥四分割（以固定种子 shuffle 后顺序切分）：**

```
全量训练集 (50,000 张, 以固定 seed 洗牌)
  │
  ├── server_val_set:   前 500 张  → 用于 GA fitness 评估
  ├── server_trust_set: 接下来 500 张 → 用于 FLTrust（v16 不使用）
  └── client_pool:      剩余 49,000 张 → Dirichlet Non-IID 分割给各客户端

全量测试集 (10,000 张) → 全量封存，仅用于每轮泛化评估和 ASR 计算
```

**断言约束：** `server_val_set ∩ client_pool = ∅`，`server_trust_set ∩ client_pool = ∅`，`server_val_set ∩ server_trust_set = ∅`

### 3.2 Non-IID 分割：Dirichlet 分布

使用 Dirichlet 分布模拟异构数据分布：

```python
# 对每个类别 c (共 10 类):
#   1. 获取属于类别 c 的所有样本索引
#   2. 从 Dir(α, α, ..., α) 采样 num_clients 维比例向量
#   3. 按比例将类别 c 的样本分配给各客户端
# α 越小 → 数据越异构 (Non-IID)
# α 越大 → 数据越均匀 (趋近 IID)
# 默认 α = 0.5
```

**算法伪代码：**

$$
\text{For each class } c \in \{0, 1, \ldots, 9\}: \quad \mathbf{p}_c \sim \text{Dir}(\alpha, \alpha, \ldots, \alpha) \in \mathbb{R}^K
$$

其中 $K$ 为客户端数量，$\alpha$ 为 Dirichlet 集中度参数。类别 $c$ 的第 $k$ 个客户端分配到的样本数为 $\lfloor p_{c,k} \cdot |D_c| \rfloor$。

---

## 4. 客户端本地训练

### 4.1 良性客户端

每轮接收全局模型参数后执行 **1 个 epoch** 的本地 SGD：

```
输入: 全局模型参数 θ_global, 本地数据 D_k
输出: 更新后的本地参数 θ_k

θ_k ← θ_global
For each mini-batch (x, y) ∈ D_k:
    ŷ = Model(x; θ_k)
    L = CrossEntropyLoss(ŷ, y)
    θ_k ← θ_k - η · ∇L

超参数: η = 0.01, momentum = 0.9 (SGD with momentum)
```

### 4.2 恶意客户端

恶意客户端首先执行与良性客户端相同的本地训练，然后由 `AttackManager` 在训练后对模型权重进行篡改。支持的攻击类型：

| 攻击类型 | 标识符 | 机制 |
|----------|--------|------|
| 标签翻转 | `label_flip` | 训练时翻转标签 |
| PGD 后门 | `backdoor` | 注入 3×3 右下角触发器 + PGD 约束 |
| Scaling | `scaling` | 放大恶意更新的模型范数 |
| 纯缩放 | `pure_scaling` | 仅放大范数，无后门训练 |
| 拜占庭 | `byzantine` | 随机扰动模型参数 |
| 模型替换后门 | `model_replacement_backdoor` | 后门训练 + 按总/恶意客户端比缩放 |

**客户端 ID 约定：** 前 `num_malicious` 个客户端为恶意客户端（ID 从 0 开始编号）。

---

## 5. 核心算法：三阶段聚合

### 5.1 Phase 1 — GA 零阶搜索（侦查）

**目标：** 搜索一个客户端权重向量 $\boldsymbol{\alpha} = (\alpha_1, \alpha_2, \ldots, \alpha_K)$，使得聚合后的模型在服务器验证集上表现最优。

#### 5.1.1 种群初始化

种群大小 $P = 15$（默认），包含三类个体：

```
种群 population ∈ ℝ^{P × K}, 其中 K = num_clients

个体 1: FedAvg 基准
    α = (1/K, 1/K, ..., 1/K)

个体 2-5: One-Hot 探针（随机探测单个/少量客户端的质量）
    随机选取 3~4 个客户端索引 I ⊂ {0, ..., K-1}
    α_i = 1/|I| if i ∈ I, else 0

剩余个体: 归一化随机向量
    α ~ Uniform(0, 1)^K, 然后归一化: α ← α / Σα_i
```

**关键设计：** 每轮全新初始化，**绝对禁止**注入任何历史轮次的权重参数。

#### 5.1.2 适应度函数

适应度函数核心思想：**Zeno 式零阶优化**——用验证集 Loss 加 L2 正则化惩罚来衡量聚合质量。

$$
\text{fitness}(\boldsymbol{\alpha}) = \frac{1}{\mathcal{L}_{\text{val}}(\boldsymbol{\theta}_{\boldsymbol{\alpha}}) + \lambda \cdot \|\boldsymbol{\theta}_{\boldsymbol{\alpha}}\|_2 + \epsilon}
$$

其中：
- $\boldsymbol{\theta}_{\boldsymbol{\alpha}} = \sum_{k=1}^{K} \alpha_k \cdot \boldsymbol{\theta}_k$ 为加权聚合后的模型参数
- $\mathcal{L}_{\text{val}}$ 为在服务器验证集上的交叉熵损失
- $\|\boldsymbol{\theta}_{\boldsymbol{\alpha}}\|_2 = \sqrt{\sum_{l} \sum_{j} \theta_{l,j}^2}$ 为所有**可训练参数**的 L2 范数
- $\lambda = 0.1$ 为正则化系数（增强对 Scaling 攻击的软门控）
- $\epsilon = 10^{-12}$ 为数值稳定性常数

**GPU 加速实现：**

```
预处理（每轮一次）:
  将所有 K 个客户端的参数展平为矩阵 M ∈ ℝ^{K × D}  (D = 总参数量)
  M[k] = flatten(θ_k)

适应度计算（每个个体）:
  α_tensor = torch.tensor(α)                    # ℝ^K
  θ_flat = α_tensor @ M                          # ℝ^D  (GPU 矩阵乘法)
  θ_params = reshape(θ_flat, param_shapes)        # 重构为模型参数
  load_state_dict(model, θ_params)                # 原地加载
  loss = CrossEntropyLoss(model(val_images), val_labels)   # GPU 前向传播
  norm = sqrt(Σ param²)                          # GPU 求范数
  return loss.item(), norm.item()
```

**异常处理：** 若 loss 或 norm 为 NaN/Inf，适应度直接返回 0.0。

#### 5.1.3 遗传操作

**每代进化流程 (共 $G = 10$ 代)：**

```
For gen = 0, 1, ..., G-1:
    1. 评估: 对种群中每个个体计算 fitness
    2. 记录全局最优: if fitness > best_fitness → 更新 best_weights
    3. 精英保留: new_pop[0] = best_weights
    4. 锦标赛选择: 从种群中选择 P 个父代
    5. 繁殖: 两两交叉 + 变异直到填满种群
```

**锦标赛选择 (Tournament Selection, k=2)：**

```
For i = 1, ..., P:
    随机抽取 2 个个体（不放回）
    选择 fitness 更高的个体作为父代
```

**线性交叉 (Linear Crossover)：**

$$
\boldsymbol{\alpha}_{\text{child}} = \beta \cdot \boldsymbol{\alpha}_{p_1} + (1-\beta) \cdot \boldsymbol{\alpha}_{p_2}, \quad \beta \sim U(0,1)
$$

然后取绝对值并归一化：$\boldsymbol{\alpha}_{\text{child}} \leftarrow \frac{|\boldsymbol{\alpha}_{\text{child}}|}{\sum |\alpha_i|}$

**高斯变异 (Gaussian Mutation)：**

$$
\text{if } u < p_{\text{mut}}: \quad \boldsymbol{\alpha} \leftarrow \boldsymbol{\alpha} + \mathcal{N}(0, \sigma^2 \mathbf{I})
$$

其中 $\sigma = 0.05$，$p_{\text{mut}} = 0.1$。变异后同样取绝对值并归一化。

**兜底机制：** 若某代进化后 `best_fitness ≤ 0`，则重新初始化整个种群。若全部 $G$ 代结束后仍无有效最优个体，则退化为 FedAvg 均匀权重 $(1/K, ..., 1/K)$。

### 5.2 Phase 2 — 锚点归一化投影（去势）

**目标：** 抵御 Scaling Attack。恶意客户端可能通过放大模型范数来主导聚合结果，锚点投影将所有客户端的参数范数统一到锚点水平。

**算法步骤：**

```
Step 1: 确定锚点
    anchor_idx = argmax(best_weights)
    // GA 赋予最高权重的客户端被认为最可信

Step 2: 计算锚点 L2 范数（仅可训练参数）
    anchor_norm = L2_norm_trainable(θ_{anchor_idx})

Step 3: 对每个客户端进行投影
    For k = 0, 1, ..., K-1:
        client_norm = L2_norm_trainable(θ_k)
        scale_k = anchor_norm / (client_norm + 1e-9)
        
        For each layer l:
            if is_trainable(l):
                θ_k[l] ← θ_k[l] × scale_k    // 缩放可训练参数
            else:
                θ_k[l] ← θ_k[l]               // BN running stats 等 buffers 原样保留
```

**数学形式化：**

$$
\hat{\boldsymbol{\theta}}_k = \boldsymbol{\theta}_k \cdot \frac{\|\boldsymbol{\theta}_{\text{anchor}}\|_2}{\|\boldsymbol{\theta}_k\|_2 + 10^{-9}}
$$

其中范数仅计算可训练参数（排除 BatchNorm 的 `running_mean`、`running_var`、`num_batches_tracked`）。

**可训练参数掩码 (trainable_mask)：** 通过比较 `model.state_dict().keys()` 和 `model.named_parameters()` 的键集合来确定哪些层参数是可训练的。

**安全告警：** 若某客户端的 `scale < 0.1`，说明其范数远大于锚点（>10倍），输出拦截警告。

### 5.3 权重融合

使用 GA 搜索得到的最优权重 $\boldsymbol{\alpha}^*$ 和投影后的参数 $\hat{\boldsymbol{\theta}}_k$ 进行加权聚合：

$$
\boldsymbol{\theta}_{\text{GA}} = \sum_{k=1}^{K} \alpha_k^* \cdot \hat{\boldsymbol{\theta}}_k
$$

**加权聚合实现细节 (`aggregate_weighted`)：**

```
For each layer l:
    if dtype(layer_l) is floating-point or complex:
        aggregated[l] = Σ_{k: α_k ≥ 1e-6} α_k · θ_k[l]
    else:
        // 非浮点层 (如 BN 的 num_batches_tracked, int64)
        // 取 α 最大的客户端的值，避免 dtype 冲突
        aggregated[l] = θ_{argmax(α)}[l]
```

### 5.4 Phase 3 — FedOpt 服务器动量平滑

**目标：** 消除全局模型在训练后期的震荡，使收敛更加平稳。

**算法（基于 FedOpt 范式）：**

```
状态变量:
    w: 全局模型参数缓冲区 (global_model_buffer)
    v: 速度缓冲区 (velocity_buffer)
    β = 0.9  (server_momentum)
    η_s = 0.3  (server_lr)

第 1 轮 (初始化):
    w ← θ_GA
    v ← 0

后续轮次 (t ≥ 2):
    δ = θ_GA - w          // 伪梯度 (pseudo-gradient)
    v ← β · v + δ         // 动量更新
    w ← w + η_s · v       // 参数更新
```

**数学形式化：**

$$
\mathbf{v}^{(t)} = \beta \cdot \mathbf{v}^{(t-1)} + (\boldsymbol{\theta}_{\text{GA}}^{(t)} - \mathbf{w}^{(t-1)})
$$

$$
\mathbf{w}^{(t)} = \mathbf{w}^{(t-1)} + \eta_s \cdot \mathbf{v}^{(t)}
$$

### 5.5 BatchNorm 校准

聚合完成后，对含 BatchNorm 层的模型（如 ResNet-20）进行 running stats 重新校准：

```
将聚合后参数加载到模型
设置 model.train() 模式
With torch.no_grad():
    For _ in range(passes):       // passes = 1
        For batch in val_images:   // batch_size = 64
            model(batch)           // 前向传播更新 running_mean/var
设置 model.eval() 模式
导出校准后的完整 state_dict（含更新后的 BN buffers）
```

**关键说明：** BN 校准只影响 `running_mean` 和 `running_var` 等 buffers，不进行反向传播，不更新可训练参数。

---

## 6. 完整聚合伪代码

```
Algorithm: VeriFL-v16 aggregate_fit(server_round, results)
────────────────────────────────────────────────────────
Input:
  results: [(client_proxy, fit_res)] × K  // K 个客户端的训练结果
  server_round: 当前轮次
  
Hyperparameters:
  P = 15        // 种群大小
  G = 10        // 进化代数
  λ = 0.1       // L2 正则化系数
  β = 0.9       // 服务器动量
  η_s = 0.3     // 服务器学习率
  k = 2         // 锦标赛大小
  σ = 0.05      // 变异标准差
  p_mut = 0.1   // 变异概率

State:
  w_buffer: 全局模型参数缓冲区 (None → 首轮初始化)
  v_buffer: 速度缓冲区 (None → 首轮初始化为 0)

────────────────────────────────────────────────────────
// Step A: GA 零阶搜索 (Phase 1)

1.  {θ_k}_{k=1}^K ← extract_ndarrays(results)
2.  GPU_preload({θ_k})                           // 展平为矩阵 M ∈ ℝ^{K×D}
3.  population ← init_population(K, P)           // 含 FedAvg + 探针 + 随机
4.  α* ← (1/K, ..., 1/K), f* ← -∞

5.  For gen = 1, ..., G:
6.      For i = 1, ..., P:
7.          f_i ← 1 / (L_val(Σ α_{i,k} θ_k) + λ·‖Σ α_{i,k} θ_k‖₂ + ε)
8.          If f_i > f*: f* ← f_i, α* ← population[i]
9.      
10.     If f* ≤ 0: population ← reinit(); continue
11.     
12.     new_pop ← [α*]                           // 精英保留
13.     parents ← tournament_select(population, scores, k=2)
14.     While |new_pop| < P:
15.         (p1, p2) ← next_pair(parents)
16.         child ← |β·p1 + (1-β)·p2|; normalize(child)     // β~U(0,1)
17.         If u < p_mut: child ← |child + N(0, σ²I)|; normalize(child)
18.         new_pop.append(child)
19.     population ← new_pop[:P]

────────────────────────────────────────────────────────
// Step B: 锚点归一化投影 (Phase 2)

20. anchor ← argmax(α*)
21. r_anchor ← ‖θ_anchor‖₂ (仅可训练参数)

22. For k = 1, ..., K:
23.     r_k ← ‖θ_k‖₂ (仅可训练参数)
24.     s_k ← r_anchor / (r_k + 1e-9)
25.     θ̂_k ← scale_trainable(θ_k, s_k)        // BN buffers 不缩放

────────────────────────────────────────────────────────
// Step C: 加权聚合

26. θ_GA ← Σ_{k=1}^K α*_k · θ̂_k

────────────────────────────────────────────────────────
// Step D: 全局动量平滑 (Phase 3)

27. If server_round == 1:
28.     w_buffer ← θ_GA
29.     v_buffer ← 0
30.     θ_final ← θ_GA
31. Else:
32.     δ ← θ_GA - w_buffer
33.     v_buffer ← β · v_buffer + δ
34.     w_buffer ← w_buffer + η_s · v_buffer
35.     θ_final ← w_buffer

────────────────────────────────────────────────────────
// Step E: BN 校准 (如有 BatchNorm)

36. If has_batchnorm:
37.     θ_final ← recalibrate_batchnorm(θ_final, val_images)

38. Return parameters(θ_final)
```

---

## 7. 模型架构

### 7.1 ResNet-20（主实验模型，含 BatchNorm）

```
ResNet-20 for CIFAR-10:
  - 输入: 3 × 32 × 32
  - Conv1: 3 → 16, kernel=3, stride=1, padding=1, bias=False
  - BN1: BatchNorm2d(16)
  - Layer1: 3 × BasicBlock(16 → 16, stride=1)
  - Layer2: 3 × BasicBlock(16 → 32, stride=2)
  - Layer3: 3 × BasicBlock(32 → 64, stride=2)
  - AvgPool2d(8)
  - Linear(64, 10)

BasicBlock(in_planes, planes, stride):
  - Conv(in_planes, planes, 3×3, stride, padding=1) → BN → ReLU
  - Conv(planes, planes, 3×3, 1, padding=1) → BN
  - Shortcut: identity 或 Conv(1×1) + BN (当 stride ≠ 1 或通道数变化)
  - 输出 = ReLU(residual + shortcut)
```

### 7.2 SimpleCNN（冒烟测试模型，无 BatchNorm）

用于快速验证功能正确性的轻量模型。

---

## 8. GPU 加速器实现细节

`GPUAccelerator` 是整个适应度评估的性能关键，设计原则为 **"脑子在 CPU，肌肉在 GPU"**。

### 8.1 初始化

```
1. 深拷贝模型模板到 GPU: model_template = deepcopy(model).to(device)
2. 预加载验证集到 GPU: val_images.to(device), val_labels.to(device)
3. 记录 state_dict 结构:
   - state_keys: 所有键名列表（含 buffers）
   - trainable_mask: 布尔列表，标记哪些是可训练参数
   - param_shapes: 各层参数形状
   - param_sizes: 各层参数扁平长度
   - total_params: 总参数量 D
4. 检测是否含 BatchNorm: has_batchnorm
```

### 8.2 客户端参数预加载

```
set_client_parameters(client_parameters):
  创建 GPU 张量: client_params_matrix ∈ ℝ^{K × D} (float32)
  For k = 0, ..., K-1:
    flat_k = concatenate(flatten(θ_k[l]) for l in layers)  // CPU
    client_params_matrix[k] = torch.tensor(flat_k, device=GPU)
```

### 8.3 适应度计算（GPU 加速）

```
calculate_fitness(α):
  α_gpu = torch.tensor(α, dtype=float32, device=GPU)
  θ_flat = α_gpu @ client_params_matrix            // ℝ^D, GPU 矩阵乘法
  θ_params = split_and_reshape(θ_flat, shapes)      // 重构为各层参数
  原地加载到 model_template
  
  model.eval()
  with no_grad:
    outputs = model(val_images)                      // GPU 批量前向
    loss = CrossEntropyLoss(outputs, val_labels)
    norm = sqrt(Σ_{p ∈ parameters} p²)
  
  return (loss.item(), norm.item())
```

**性能优势：** 将 $P \times G = 150$ 次适应度评估从 CPU 串行计算转为 GPU 矩阵运算 + 批量前向传播。

---

## 9. 评估指标

每轮在**封存测试集**上计算（不参与 GA 搜索）：

| 指标 | 计算方式 |
|------|----------|
| **Test Loss** | $\frac{1}{|B|} \sum_{b \in B} \text{CrossEntropy}(\hat{y}_b, y_b)$ |
| **Test Accuracy** | $\frac{\text{correct predictions}}{|D_{\text{test}}|}$ |
| **ASR** (后门攻击时) | 在非目标类样本上注入 3×3 右下角触发器（值=1.0），计算被预测为目标类的比例 |
| **Round Time** | 单轮实际用时（秒） |
| **Total Time** | 累计运行时间（秒） |

**ASR 计算细节：**
```
1. 过滤掉原始标签 == target_label 的样本
2. 在 clean_images 的右下角 3×3 区域注入全 1 触发器
3. ASR = Σ(predicted == target_label) / total_backdoor
```

---

## 10. 超参数汇总

### 10.1 算法超参数

| 参数 | 符号 | 默认值 | 说明 |
|------|------|--------|------|
| 种群大小 | $P$ | 15 | GA 种群个体数量 |
| 进化代数 | $G$ | 10 | 每轮聚合的 GA 进化代数 |
| L2 正则化系数 | $\lambda$ | 0.1 | 适应度函数中的模型范数惩罚 |
| 服务器动量 | $\beta$ | 0.9 | FedOpt EMA 动量系数 |
| 服务器学习率 | $\eta_s$ | 0.3 | FedOpt 步长 |
| 锦标赛大小 | $k$ | 2 | 锦标赛选择的参与者数 |
| 变异标准差 | $\sigma$ | 0.05 | 高斯变异噪声大小 |
| 变异概率 | $p_{\text{mut}}$ | 0.1 | 执行变异的概率 |
| 数值稳定常数 | $\epsilon$ | $10^{-12}$ | 避免除零 |

### 10.2 训练超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 客户端优化器 | SGD(lr=0.01, momentum=0.9) | 所有客户端统一 |
| 客户端本地 Epoch | 1 | 每轮训练 1 个 epoch |
| 批大小 | auto (256 for ≥20GB VRAM, 128 for ≥10GB, 64 otherwise) | 自适应 GPU 显存 |
| 总轮次 | 500 | 默认实验轮数 |
| 随机种子 | [42, 2024, 3407] | 多种子重复实验 |

### 10.3 联邦学习配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 客户端总数 | 10 | 全参与（无采样） |
| 恶意客户端数 | 1-4 (取决于场景) | 占比 10%-40% |
| Non-IID α | 0.5 | Dirichlet 集中度 |
| 服务器验证集 | 500 条（从训练集切分） | 用于 GA fitness |
| 服务器可信集 | 500 条（从训练集切分） | FLTrust 用，v16 不使用 |

---

## 11. 配置文件格式

### 11.1 场景配置 (scenario.yaml)

```yaml
scenario_name: "cifar10_backdoor_40pct"
data:
  dataset: CIFAR10
  alpha: 0.5
  batch_size: auto
model:
  name: ResNet20
attack:
  type: backdoor           # none | label_flip | backdoor | scaling | ...
  num_malicious: 4
  target_label: 0
  params:
    poison_ratio: 0.5
    trigger_size: 3
    eps: 2.0
simulation:
  num_clients: 10
  rounds: 500
  seeds: [42, 2024, 3407]
server:
  val_ratio: 500
  trust_ratio: 500
```

### 11.2 方法配置 (method.yaml)

```yaml
strategies:
  - name: "v16"
    enabled: true
    params:
      pop_size: 15
      generations: 10
      lambda_reg: 0.1
```

### 11.3 运行命令

```bash
python src/main.py \
  --scenario configs/scenarios/cifar10_backdoor.yaml \
  --method configs/methods/v16.yaml
```

---

## 12. 结果存储

```
results/
  └── CIFAR10_ResNet20/                           # env_name = {dataset}_{model}
      └── mc10_backdoor_mr0.3_a0.5/              # scenario_name
          └── v16_ps15_g10/                       # strategy_dir_name
              └── seed_42/
                  ├── metrics.csv                 # 每轮指标
                  └── config.json                 # 实验配置快照
```

**metrics.csv 列：** round, loss, accuracy, asr (可选), round_time, total_time

---

## 13. 复现检查清单

1. **环境准备**
   - Python 3.8+, PyTorch, Flower (flwr), Ray, torchvision, numpy, pyyaml
   - GPU（推荐 ≥12GB VRAM）

2. **数据准备**
   - CIFAR-10 数据集（自动下载至 data）

3. **关键实现验证**
   - [ ] 种群初始化包含 FedAvg + One-Hot 探针 + 随机向量
   - [ ] 适应度函数 = 1/(Loss + λ·Norm + ε)，λ=0.1
   - [ ] 锚点为 GA 赋权最大的客户端
   - [ ] 投影仅缩放可训练参数，BN buffers 保持不变
   - [ ] 服务器动量 β=0.9, η_s=0.3，第一轮初始化 velocity=0
   - [ ] BN 校准使用验证集前向传播刷新 running stats
   - [ ] 验证集、可信集、客户端数据互斥
   - [ ] 恶意客户端为前 num_malicious 个

4. **快速验证**

   ```bash
   # 导入测试
   python -c "from src.strategies.ours.v16 import StrategyV16; print('OK')"
   
   # 冒烟测试 (3 轮, 3 客户端)
   python src/main.py \
     --scenario configs/scenarios/smoke_test.yaml \
     --method configs/methods/v16.yaml
   ```