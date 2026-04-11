# ShieldFL 实验平台基础设施审计报告

> **日期**：2025-07-08  
> **审计范围**：ShieldFL 全平台——聚合管线、攻击/防御框架、配置传播、数据加载、评估管线、确定性控制、BN 处理、服务端学习动力学  
> **起点**：Scaling Attack 复现失败（AC-13 FAIL，模型崩溃）  
> **目标**：以复现失败为线索，对所有可能影响攻防复现正确性的基础设施问题进行地毯式排查  

---

## 0. 审计方法论

以 Scaling Attack 复现失败的两个致命根因（γ=N 公式误用、VeriFL ≠ FedAvg）为切入点，逆向追踪每个根因所涉及的子系统，再从每个子系统出发做完整审计。共覆盖 **9 个子系统**，审计了 **30+ 源文件**，产出 **28 项发现**。

```
Scaling 失败 → 聚合管线（VeriFL 4 阶段 vs FedAvg）
            → 攻击框架（4 类攻击分发）
            → 防御框架（17 个常量 / 16 个分支）
            → 配置传播（YAML → args → 代码链路）
            → 数据加载（Dirichlet 分区 + 样本截断）
            → 评估管线（Clean Acc + ASR 计算）
            → 确定性控制（种子传播 + 组件隔离）
            → BN 处理（全流程 running_mean/var 追踪）
            → 服务端动力学（momentum + lr + GA + L2 投影）
```

---

## 1. 发现清单总览

| 严重级 | 编号 | 简述 | 影响范围 |
|--------|------|------|----------|
| 🔴 致命 | F-1 | VeriFL `on_before_aggregation` 跳过所有 FedML 原生防御 | 所有防御实验 |
| 🔴 致命 | F-2 | Trimmed Mean 是假实现（`compute_a_score` = `return sample_num`） | Trimmed Mean 防御 |
| 🔴 致命 | F-3 | Bulyan 未注册到 FedMLDefender 且缺少接口方法 | Bulyan 防御 |
| 🔴 致命 | F-4 | VeriFL 完全绕过 `FedMLAggOperator.agg()`，非 FedAvg 聚合 | VeriFL 下所有实验 |
| 🔴 致命 | F-5 | VeriFL 丢弃 `sample_num`，YAML 的 `FedAvg` 标注与实际行为矛盾 | VeriFL 下所有实验 |
| 🟠 严重 | S-1 | Server momentum 无差别应用于 BN buffers 和 `num_batches_tracked` | VeriFL 下所有含 BN 模型 |
| 🟠 严重 | S-2 | DP 链断裂：`on_before_aggregation` 跳过 clip，`on_after_aggregation` 可能注入 noise | VeriFL + DP 场景 |
| 🟠 严重 | S-3 | Baseline 路径无 BN recalibration，与 VeriFL 对比不公平 | VeriFL vs Baseline 实验 |
| 🟠 严重 | S-4 | L2 投影不保护 BN buffers，攻击者 BN 统计量不受约束 | VeriFL + 任意模型攻击 |
| 🟠 严重 | S-5 | GA fitness 目标与实际聚合结果不对齐（无投影+无 recal） | VeriFL 下所有实验 |
| 🟡 中等 | M-1 | YAML 命名空间平铺，`train_args` 与 `shieldfl_args` 可能冲突 | 配置维护 |
| 🟡 中等 | M-2 | `verifl_aggregator.py` L69 强制覆写 `args.aggregator_type` | 配置语义 |
| 🟡 中等 | M-3 | `FedMLAggOperator.agg()` 原地修改 client 0 的 `state_dict` | Baseline 路径 |
| 🟡 中等 | M-4 | `sort_client_updates` 是死标志，无实际排序逻辑 | 确定性保证的可信度 |
| 🟡 中等 | M-5 | GPUAccelerator `recalibrate_batchnorm` 重置全局 PyTorch 种子 | VeriFL 确定性 |
| 🟡 中等 | M-6 | BN recalibration 仅 1 pass，可能不充分收敛 | VeriFL 下含 BN 模型 |
| 🟡 中等 | M-7 | `FedMLAggOperator` FedOpt 分支为 `pass`（空操作） | FedOpt 场景 |
| 🟡 中等 | M-8 | `on_after_aggregation` 未覆写可能在 BN recal 后叠加 DP 噪声 | VeriFL + DP |
| 🟡 中等 | M-9 | FedMLDefender 硬编码 if-elif 分发，无注册机制 | 扩展性 |
| 🟡 中等 | M-10 | 默认 `server_lr=0.3` + `server_momentum=0.9` 大幅减缓收敛 | VeriFL 学习动力学 |
| 🔵 低 | L-1 | `ByzantineAttack` 使用全局 RNG 无隔离 | 确定性（当前不触发） |
| 🔵 低 | L-2 | FedML 客户端子采样用 `np.random.seed(round_idx)` 污染全局状态 | 部分参与场景 |
| 🔵 低 | L-3 | cclip/wbc 防御使用全局 RNG | 这些防御场景 |
| 🔵 低 | L-4 | `num_workers>0` 时缺少 `worker_init_fn` | 多进程数据加载 |
| 🔵 低 | L-5 | `trust_loader` 已创建但未被任何代码使用 | 资源浪费 |
| ⚪ 信息 | I-1 | `federated_optimizer: "FedAvg"` 仅为管线入口门控 | 文档 |
| ⚪ 信息 | I-2 | ASR 评估对全量图片注入触发器但只统计子集 | 学术报告注意事项 |
| ⚪ 信息 | I-3 | `val_global == test_global` 在 FedML 框架中 | 无 early-stopping 意义 |

---

## 2. 致命问题详解（🔴 Fatal）

### F-1：VeriFL `on_before_aggregation` 跳过所有 FedML 原生防御

**文件**：[verifl_aggregator.py L95-113](python/examples/federate/prebuilt_jobs/shieldfl/trainer/verifl_aggregator.py)

**证据**：`ServerAggregator` 基类的 `on_before_aggregation()` 包含 **6 步**完整的安全链：

```
① FHE 解密 → ② Global DP clip → ③ 数据重建攻击 → ④ 模型攻击 → ⑤ FedMLDefender → ⑥ benign_client_idxs
```

VeriFL 覆写后**仅保留第④步**（模型攻击 hook），**跳过第②⑤⑥步**：

```python
# VeriFL 的 on_before_aggregation（精简）
def on_before_aggregation(self, ...):
    if FedMLAttacker.get_instance().is_model_attack():
        raw_client_model_or_grad_list = FedMLAttacker.get_instance().attack_model(...)
    return raw_client_model_or_grad_list
```

**影响**：

| 被跳过的步骤 | 后果 |
|-------------|------|
| Global DP clip | `defense_type=norm_diff_clipping` 等需要梯度裁剪的防御完全失效 |
| FedMLDefender | **所有 17 种已注册防御**（Krum、多Krum、Trimmed Mean、Median、FoolsGold、CRFL、RFA 等）**在 VeriFL 路径下全部无效** |
| `benign_client_idxs` | Defender 输出的良性客户端过滤列表被忽略 |

**验证**：历史 193 组实验全部使用 `defense_type=none`，因此此 bug 尚未被触发。但任何未来尝试在 VeriFL 下启用防御的实验都将静默失败——**防御不报错但不起作用**。

**结论**：这不是 bug，而是 VeriFL 的**设计性取舍**——它用 GA + L2 投影替代了 FedML 原生防御。但这一取舍**未被文档化**，且与 YAML 中 `enable_defense` / `defense_type` 参数的存在相矛盾，容易导致研究者误以为防御已启用。

---

### F-2：Trimmed Mean 是假实现

**文件**：[trimmed_mean_defense.py](python/fedml/core/security/defense/trimmed_mean_defense.py) → [common/utils.py](python/fedml/core/security/common/utils.py)

**证据链**：

```python
# trimmed_mean_defense.py
def defend_before_aggregation(self, ...):
    ...
    score_list = self._compute_a_score(client_grad_list)
    importance_feature_list = self._get_importance_feature(score_list)
    ...

# common/utils.py
def compute_a_score(local_updates_list, ...):
    score_list = []
    for item in local_updates_list:
        (local_sample_number, ...) = item
        score_list.append(local_sample_number)  # ← 直接返回样本数！
    return score_list  # todo
```

**实际行为**：所谓 "Trimmed Mean" 其实是**按客户端样本数排序后截断**——不是坐标级 coordinate-wise trimmed mean。`# todo` 注释确认这是一个未完成的占位符。

**正确的 Trimmed Mean**：对每个模型参数的每个坐标，收集所有客户端的该坐标值，排序后去掉最大最小 β 比例，取均值。FedML 的 `BulyanDefense` 内部反而有正确的 `trimmed_mean()` 静态方法实现。

**影响**：任何使用 `defense_type=trimmed_mean` 的实验所报告的防御效果都不可信。

---

### F-3：Bulyan 未注册且接口不兼容

**文件**：[fedml_defender.py](python/fedml/core/security/fedml_defender.py)、[bulyan_defense.py](python/fedml/core/security/defense/bulyan_defense.py)

**证据**：

1. **未注册**：`FedMLDefender.init()` 的 if-elif 链中有 17 个 `DEFENSE_*` 常量，但**没有 `bulyan` 分支**。设置 `defense_type=bulyan` 将触发 `raise Exception("args.defense_type is not defined")`。

2. **接口不兼容**：`BulyanDefense` 只实现了 `run()` 方法。FedMLDefender 分发通过 `defend_before_aggregation()` / `defend_on_aggregation()` / `defend_after_aggregation()` 三个 hook 调用。即使注册了 Bulyan，也会因缺少这些方法而失败。

3. **内部实现反而正确**：讽刺的是，`BulyanDefense` 内部正确实现了 coordinate-wise trimmed mean（Bulyan 的第二阶段）和 Krum 选择（第一阶段），算法本身没有问题。

**影响**：Bulyan 防御完全不可用。

---

### F-4：VeriFL 完全绕过 `FedMLAggOperator.agg()`

**文件**：[verifl_aggregator.py L134-252](python/examples/federate/prebuilt_jobs/shieldfl/trainer/verifl_aggregator.py)、[agg_operator.py](python/fedml/ml/aggregator/agg_operator.py)

**证据**：

`VeriFL.aggregate()` 自行实现 4 阶段聚合流程：
```
Phase 1: MicroGA 进化搜索 α 权重 → 验证集 loss 最优
Phase 2: L2 范数投影到锚点客户端
Phase 3: 服务器动量 SGD 更新
Phase 4: BN 重校准
```

整个过程**不调用 `super().aggregate()`**，也不调用 `FedMLAggOperator.agg()`。

**对比标准 FedAvg**：

| 维度 | 标准 FedAvg | VeriFL |
|------|-----------|--------|
| 聚合权重 | $w_i = n_i / \Sigma n$ （样本量） | GA 搜索得到的 $\alpha_i$（验证集 fitness） |
| 范数约束 | 无 | L2 投影到锚点 |
| 服务端优化器 | 无 | Momentum SGD ($\mu$=0.9, $\eta_s$=0.3) |
| BN 处理 | 加权平均 | Recalibration via validation forward pass |
| FedMLDefender 集成 | 通过 `defend_on_aggregation` hook | **完全绕过** |

**影响**：YAML 中 `federated_optimizer: "FedAvg"` 给人以 FedAvg 语义的印象，但 VeriFL 路径下的聚合行为与 FedAvg **完全不同**。这直接导致：
- Scaling Attack 的 γ=N 替换公式设计假设被违反
- 攻防论文中的理论分析（假设 FedAvg 加权平均）不适用
- 实验结果不能直接与标准 FL 文献对比

---

### F-5：VeriFL 静默丢弃 `sample_num`

**文件**：[verifl_aggregator.py L145-148](python/examples/federate/prebuilt_jobs/shieldfl/trainer/verifl_aggregator.py)

**证据**：

```python
weights_results = [
    self._ordered_dict_to_ndarrays(client_state)
    for _, client_state in raw_client_model_or_grad_list  # ← _ 丢弃 sample_num
]
```

`fedml_aggregator.py` 构建的 `model_list` 每个元素是 `(sample_num, state_dict)` 的 tuple。VeriFL 在解构时直接丢弃 `sample_num`。

**影响**：在 non-IID 设定（如 α=0.1）下，客户端数据量可能差异达 6 倍（50 vs 300）。标准 FedAvg 通过 `n_i/Σn` 加权来反映数据量差异；VeriFL 完全忽略此信息。这使得 non-IID 实验的聚合行为与文献不可比较。

---

## 3. 严重问题详解（🟠 Severe）

### S-1：Server Momentum 无差别作用于 BN buffers

**文件**：[verifl_aggregator.py L224-235](python/examples/federate/prebuilt_jobs/shieldfl/trainer/verifl_aggregator.py)

Phase 3 的 momentum 更新循环遍历 **ALL** numpy arrays，不使用 `trainable_mask`：

$$v^{(t)}_{\text{ALL}} = \mu \cdot v^{(t-1)}_{\text{ALL}} + (\hat{\theta}_{\text{ALL}} - \theta^{(t)}_{\text{ALL}})$$

这导致：
- `running_mean`、`running_var`：被动量平滑化处理（语义错误但被 Phase 4 recalibration 掩盖）
- `num_batches_tracked`（int64）：经过 `0.9 * int + int` 运算后变为浮点，再被强制转回 int64，值失去意义

**当前影响**：被 Phase 4 BN recalibration 掩盖。但若 recalibration 因任何原因失效（`has_batchnorm=False` 误判、val 数据为空、模型不含 BN），Bug 将暴露。

---

### S-2：DP 链断裂

**证据链**：

1. `on_before_aggregation`（VeriFL 覆写）：**跳过** Global DP clip
2. `on_after_aggregation`（**未覆写**，走基类）：可能执行 Global DP noise 注入

结果：如果同时启用 VeriFL + DP，噪声被注入但裁剪缺失 → DP 保证不成立（$(\epsilon, \delta)$-DP 需要先 clip 再加 noise）。

**当前影响**：未观察到启用 DP 的实验。但配置层面不阻止这种组合。

---

### S-3：Baseline 与 VeriFL 的 BN 处理不对齐

| 维度 | VeriFL | Baseline (FedAvg) |
|------|--------|-------------------|
| BN 聚合 | GA 线性加权（Phase 1） | 按 `sample_num` 加权平均 |
| BN 投影 | Phase 2 **不投影** BN | 无投影 |
| BN momentum | Phase 3 错误地应用 momentum | 无 |
| BN recalibration | Phase 4 ✅ 在 val 集上重校准 | ❌ **无** |
| 服务端 momentum | ✅ | ❌ |

**影响**：当对比 `aggregator_type=verifl` 与 `aggregator_type=fedavg` 的实验结果时，差异不仅来自 GA 权重搜索，还来自 BN recalibration 和 server momentum。Baseline 路径缺少 recalibration 在 non-IID 场景下会导致 BN 统计量偏移，使其 accuracy 偏低，从而**人为抬高 VeriFL 的相对优势**。

---

### S-4：L2 投影不保护 BN buffers

**文件**：[gpu_accelerator.py L20-21](python/examples/federate/prebuilt_jobs/shieldfl/trainer/gpu_accelerator.py)

```python
trainable_mask = [k in named_parameters_keys for k in state_keys]
# BN running_mean, running_var, num_batches_tracked → False → 不被投影
```

投影公式：

$$\text{scale}_j = \frac{\|\theta_{\text{anchor}}\|_2}{\|\theta_j\|_2 + \epsilon}, \quad \theta_j^{\text{proj}} = \text{scale}_j \cdot \theta_j \quad \text{(仅 trainable)}$$

恶意客户端可训练参数被缩放到锚点范数，但 BN buffers **直接透传**。Scaling Attack 的恶意训练改变了特征分布 → `running_mean`/`running_var` 偏移 → 加权聚合后污染全局 BN → Phase 4 recalibration 需要从偏移的初始值开始恢复。

**影响**：BN 是攻击者可利用的未保护信道。

---

### S-5：GA Fitness 与实际聚合不对齐

**GA 优化目标**：

```python
fitness(α) = validation_loss(model_loaded_with(α · client_params)) + λ · norm_penalty
```

GA 评估的模型是 **Phase 1 直接加权**的结果——**未经 Phase 2 L2 投影**，**未经 Phase 4 BN recalibration**。

但最终聚合使用 GA 找到的 $\alpha^*$ 后，还要经过投影 + momentum + recal。这意味着 GA 优化的目标函数与实际 deploy 的模型之间存在 **系统性偏差**：

$$\text{GA 优化}: \mathcal{L}(\sum_i \alpha_i \theta_i) \neq \mathcal{L}(\text{momentum}(\text{project}(\sum_i \alpha_i \theta_i)))$$

**影响**：GA 找到的 "最优" 权重可能在经过后续处理后不再最优。尤其在有攻击者时，GA 可能给恶意客户端较低权重（因为直接加权后 val loss 高），但 L2 投影已经缩放了恶意参数，投影后的 val loss 可能更低——GA 无法感知 this。

---

## 4. 中等问题详解（🟡 Moderate）

### M-1：YAML 命名空间平铺

**文件**：[arguments.py L189-191](python/fedml/arguments.py)

```python
def set_attr_from_config(self, configuration):
    for _, param_family in configuration.items():
        for key, val in param_family.items():
            setattr(self, key, val)
```

所有 YAML section（`train_args`、`shieldfl_args`、`data_args` 等）的 key 被 `setattr` 到同一个 `args` 对象。若两个 section 出现同名 key，后面的 section **静默覆盖**前面的，取决于 YAML 字典键遍历顺序。

**当前风险**：`train_args.server_lr` 与假设的 `shieldfl_args.server_lr` 若同时存在会冲突。目前尚未发生。

---

### M-2：`aggregator_type` 强制覆写

**文件**：[verifl_aggregator.py L69](python/examples/federate/prebuilt_jobs/shieldfl/trainer/verifl_aggregator.py)

```python
setattr(args, "aggregator_type", "verifl")
```

VeriFL 在 `__init__` 中强制将 `aggregator_type` 改为 `"verifl"`，即使 YAML 中设置了其他值。这意味着：

- 一旦代码路径进入 VeriFL aggregator，`args.aggregator_type` 的 YAML 原始值被覆盖
- `MetricsCollector` 日志中记录的 `aggregator` 字段始终为 `"verifl"` 而非 YAML 配置值
- 外部脚本若读取 `args.aggregator_type` 会得到被覆写后的值

---

### M-3：FedAvg `agg()` 原地修改 client 0

**文件**：[agg_operator.py L38-44](python/fedml/ml/aggregator/agg_operator.py)

```python
(num0, avg_params) = raw_grad_list[0]   # ← 引用，不是 copy！
for k in avg_params.keys():
    for i in range(len(raw_grad_list)):
        ...
        avg_params[k] += local_model_params[k] * w   # i=0 时自乘 w 后覆盖
```

`avg_params` 是 client 0 的 `state_dict` 的**引用**。i=0 时 `avg_params[k] += ... * w` 将 client 0 的原始参数乘以 $w_0$ 而非保持原样。后续 i=1..N-1 的累加基于已被修改的 `avg_params[k]`。

**但 i=0 的处理**：`avg_params[k] = avg_params[k] * w` 在第一次循环将值设为 `client0_param * w0`，后续 `+= client_i_param * w_i`，最终结果 = $\sum_i w_i \cdot \theta_i$，**数学上正确**。问题是 client 0 的 `model_dict` 对象被 in-place 修改了。

**影响**：在 Baseline 路径下，`model_dict[0]` 在聚合后被改变。如果后续代码（如防御、日志）再读取 `model_dict[0]` 的原始参数，会得到聚合后的值。VeriFL 路径不经过此代码。

---

### M-4：`sort_client_updates` 死标志

**文件**：[verifl_aggregator.py L107](python/examples/federate/prebuilt_jobs/shieldfl/trainer/verifl_aggregator.py)

```python
logging.info(f"  Sort client updates: {self.sort_client_updates}")
```

仅用于日志打印。实际客户端遍历顺序由 `range(self.client_num)` 决定，天然是 0..N-1 固定顺序。该标志给人以"确定性排序已启用"的假象但无实际效果。

---

### M-5：GPUAccelerator 重置全局 PyTorch 种子

**文件**：[gpu_accelerator.py L56](python/examples/federate/prebuilt_jobs/shieldfl/trainer/gpu_accelerator.py)

```python
torch.manual_seed(self.seed)  # 每轮 BN recal 重置全局 torch RNG
```

这在每轮聚合的 Phase 4 将全局 PyTorch RNG 回退到初始种子。后续依赖全局 torch RNG 的操作（如 Dropout、随机初始化）的随机状态被"倒回"。

**当前影响**：ShieldFL 使用的模型（SimpleCNN、ResNet18、LeNet5）**不含 Dropout**，因此训练不受影响。但这是一个维护隐患。

**修复**：改用 `torch.Generator` 隔离 RNG。

---

### M-6：BN Recalibration 仅 1 pass

**文件**：[gpu_accelerator.py L52-66](python/examples/federate/prebuilt_jobs/shieldfl/trainer/gpu_accelerator.py)

`passes=1`，对 validation set 仅做一次前向传播。PyTorch BN 默认 momentum=0.1，意味着 `running_mean = 0.9 * old + 0.1 * batch_mean`。

若 Phase 3 的 momentum 污染导致 BN stats 严重偏移，1 pass 后:
- 新值 = 0.9 × 污染值 + 0.1 × 真实统计 → **仍保留 90% 的污染**

多 pass（如 5-10）可以更好地洗掉污染。这在有攻击的场景下尤为重要。

---

### M-7：FedOpt 分支空实现

**文件**：[agg_operator.py](python/fedml/ml/aggregator/agg_operator.py)

```python
elif args.federated_optimizer == "FedOpt":
    pass  # 无操作
```

如果有人将 `federated_optimizer` 设为 `"FedOpt"`，聚合阶段不执行任何操作，返回 `None`，下游代码将崩溃。

---

### M-8：`on_after_aggregation` 在 BN recal 后叠加处理

VeriFL 的 `aggregate()` 在返回前完成 BN recalibration。但 `on_after_aggregation`（基类）在 `aggregate()` 返回后被调用，可能注入 DP 噪声或执行 defense 后处理。这些操作修改权重参数但**不重新校准 BN**，导致 BN stats 与修改后的权重不匹配。

---

### M-9：FedMLDefender 硬编码 if-elif

**文件**：[fedml_defender.py](python/fedml/core/security/fedml_defender.py)

17 个 defense 常量、16 个 if-elif 分支，没有注册表（registry）机制。新增防御需要：
1. 在文件顶部添加常量
2. 在 `init()` 中添加 elif 分支
3. 确保防御类实现正确的 hook 方法

`BulyanDefense` 就是因为缺少第 2 步而无法使用的例子。

---

### M-10：默认 server_lr=0.3 + server_momentum=0.9 的动力学影响

VeriFL 默认参数（从 YAML 和代码审计确认）：

$$v^{(t)} = 0.9 \cdot v^{(t-1)} + \Delta^{(t)}, \quad \theta^{(t+1)} = \theta^{(t)} + 0.3 \cdot v^{(t)}$$

**第一轮**（$v^{(-1)}=0$）：
$$\theta^{(1)} = \theta^{(0)} + 0.3 \cdot \Delta^{(0)}$$

全局模型仅朝聚合方向移动 **30%**。这显著减缓了初始收敛，尤其在攻防实验中增加了不确定性——攻击者在前期可能因模型收敛慢而表现出不同于文献的行为。

**注意**：YAML 中 `server_momentum: 0.0` 和 `server_lr` 默认值需要与 VeriFL 代码实际使用的默认值交叉验证。YAML 写 `server_momentum: 0.0`，但 VeriFL 代码中 `getattr(args, 'server_momentum', 0.9)` 的 fallback 为 0.9——若 YAML 成功传播则为 0.0，否则 fallback 为 0.9。

---

## 5. 低级与信息性问题（🔵🟢 / ⚪）

### L-1 ~ L-4：确定性缺口

| 编号 | 问题 | 触发条件 | 当前影响 |
|------|------|----------|----------|
| L-1 | `ByzantineAttack` 用全局 Python/numpy RNG | `attack_type=byzantine` | 不触发（ShieldFL 用 label_flipping/model_replacement） |
| L-2 | FedML `np.random.seed(round_idx)` 污染全局 numpy | `client_num_per_round < total` | 不触发（ShieldFL 全员参与） |
| L-3 | cclip/wbc 防御用全局 numpy RNG | 启用这些防御 | 不触发（当前 defense_type=none） |
| L-4 | DataLoader 缺 `worker_init_fn` | `num_workers > 0` | 不触发（当前 num_workers=0） |

### I-1：`federated_optimizer: "FedAvg"` 的真实含义

`federated_optimizer` **不控制聚合算法**。它仅是 cross_silo 管线入口的门控条件：

```python
if args.federated_optimizer == "FedAvg":
    from fedml.cross_silo.server import server_initializer
    server_initializer.init_server(args, ..., server_aggregator)
```

必须为 `"FedAvg"` 才能进入标准 MPI 消息循环。实际聚合由 `shieldfl_args.aggregator_type`（`"verifl"` 或 `"fedavg"`）在 `main_fedml_shieldfl.py` 中分支决定。

---

## 6. 跨子系统影响分析

### 6.1 Scaling Attack 失败的完整因果链

```
YAML: federated_optimizer="FedAvg", aggregator_type="verifl"
  ↓
main_fedml_shieldfl.py: 创建 VeriFL aggregator（F-4: 绕过 FedAvg）
  ↓
verifl_aggregator.__init__: setattr(args, "aggregator_type", "verifl")（M-2: 强制覆写）
  ↓
客户端训练: γ=10 放大 → ‖θ_m‖ ≈ 10·‖θ_b‖（BN stats 也被恶意训练改变）
  ↓
Phase 1 (GA): fitness 在未投影模型上评估（S-5: 目标不对齐）
  → GA 可能给恶意客户端较低 α（因 val loss 高）
  → 但也可能给较高 α（因放大后模型在某些字段表现好）
  → BN stats 被线性加权组合（语义不完全正确）
  ↓
Phase 2 (L2): 可训练参数被投影到锚范数（削弱 γ 放大）
  → BN buffers 不投影（S-4: BN 漏洞）
  ↓
Phase 3 (Momentum): 30% 步长 × 动量更新（M-10: 减缓攻击效果）
  → BN 被错误地应用 momentum（S-1）
  ↓  
Phase 4 (BN recal): 覆盖 Phase 3 的 BN 损坏
  → 但 1 pass 可能不够（M-6）
  → 全局种子被重置（M-5）
  ↓
γ=10 + K=3 → 有效放大 Kγ/N = 3 → 模型崩溃
```

**总结**：Scaling Attack 失败是**至少 7 个基础设施因素**叠加的结果，而非单一 bug。

### 6.2 对未来 FLTrust 复现的影响

FLTrust 需要：
1. 服务端用 trust dataset 训练 trust model → **trust_loader 已创建但未使用**（L-5）
2. 余弦相似度 + trust score 加权聚合 → 需绕过 VeriFL 的 GA 机制或在 Baseline 基础上实现
3. ReLU 裁剪到与 trust model 方向一致 → 需在 FedMLDefender 注册或自行实现
4. FedAvg 加权平均 → Baseline 路径可用但缺少 BN recalibration（S-3）

**风险**：如果在 VeriFL 路径下实现 FLTrust 逻辑，FedMLDefender 不可用（F-1）；如果在 Baseline 路径下实现，BN 处理缺失（S-3）。建议在 Baseline 基础上补充 BN recalibration 或实现独立的 FLTrust aggregator。

### 6.3 对已有 193 组实验的回溯评估

| 实验类型 | 受影响的问题 | 结果可信度 |
|----------|-------------|-----------|
| `defense_type=none, aggregator=fedavg` | M-3（client 0 原地修改） | ✅ 高（数学结果正确，副作用不影响聚合） |
| `defense_type=none, aggregator=verifl` | F-4, F-5, S-1, S-5, M-2, M-5, M-6, M-10 | ⚠️ **中**（结果自洽但不代表标准 FedAvg） |
| `defense_type=trimmed_mean` | F-2 | ❌ **不可信**（假实现） |
| `defense_type=bulyan` | F-3 | ❌ **无法运行** |
| VeriFL + 任何防御 | F-1 | ❌ **防御静默失效** |

---

## 7. 修复优先级路线图

### P0（立即修复 — 阻塞所有防御实验）

| 编号 | 修复内容 | 预估工作量 |
|------|---------|-----------|
| F-1 | 在 VeriFL `on_before_aggregation` 中恢复 FedMLDefender 调用，或明确文档化该设计决策 | 小 |
| F-2 | 实现真正的 coordinate-wise trimmed mean（可从 `BulyanDefense` 移植） | 中 |
| F-3 | 在 FedMLDefender 注册 Bulyan + 补充 `defend_before_aggregation()` 方法 | 中 |

### P1（高优先级 — 影响实验正确性）

| 编号 | 修复内容 | 预估工作量 |
|------|---------|-----------|
| S-1 | Phase 3 momentum 更新使用 `trainable_mask` 跳过 BN buffers | 小 |
| S-3 | Baseline 路径补充可选的 BN recalibration | 中 |
| S-5 | GA fitness 评估时增加投影 + recalibration 使目标函数对齐 | 大 |
| M-6 | BN recalibration `passes` 参数化（建议默认 5-10）| 小 |

### P2（中优先级 — 改善可维护性和诊断性）

| 编号 | 修复内容 |
|------|---------|
| M-1 | 添加 YAML namespace collision 检测 |
| M-2 | 移除强制覆写，让 YAML 值透传并在日志中记录原始值 |
| M-5 | GPUAccelerator 改用 `torch.Generator` 隔离 RNG |
| M-9 | 防御分发改为 registry pattern |
| S-2 | VeriFL 覆写 `on_after_aggregation` 确保 DP/defense 后处理的正确排序 |

### P3（低优先级 — 防御性改进）

| 编号 | 修复内容 |
|------|---------|
| L-1 ~ L-4 | 各组件 RNG 隔离、worker_init_fn 添加 |
| M-4 | 移除或实现 `sort_client_updates` |
| M-7 | FedOpt 分支实现或移除 |

---

## 8. 数据依赖总结

### 8.1 Dirichlet 数据分配与样本截断

| α | 每 client 样本数（5 client, max=300） | 分布特征 |
|---|------|----------|
| 0.1 | ~50 – 300 | 极度不均，部分 client 远低于上限 |
| 0.5 | ~150 – 300 | 中度不均 |
| 100 | ~300 – 300 | 近似 IID，全部命中上限 |

`max_samples_per_client=300` 是实际均衡机制。FedAvg 权重 $w_i = n_i / \Sigma n$ 在低 α 下有显著差异；VeriFL 因丢弃 `sample_num` 而无视此差异（F-5）。

### 8.2 评估管线状态

| 检查项 | 状态 |
|--------|------|
| 测试在所有聚合阶段完成后执行 | ✅ |
| ASR 在全局模型上评估 | ✅ |
| 触发器参数训练/评估一致 | ✅ |
| 三路数据（val/trust/test）不重叠 | ✅ |
| 每轮 round_idx 无 off-by-one | ✅ |
| 被评估模型 = 将下发给客户端的模型 | ✅ |
| JSONL 指标输出完整 | ✅ |

### 8.3 确定性保证矩阵

| 场景 | 可复现性 |
|------|----------|
| CPU + FedAvg + 无攻防 | ✅ 完全确定 |
| GPU + FedAvg + label_flipping | ✅ 预期确定（攻击用隔离 RNG） |
| GPU + VeriFL + 任意攻击 | ⚠️ 近似确定（M-5: BN recal 重置全局种子） |
| 启用 cclip/wbc 防御 | ⚠️ 可能不完全确定（L-3） |
| `num_workers > 0` | ❌ 不确定（L-4） |
| 部分客户端参与 | ⚠️ 全局 numpy 被污染（L-2） |

---

## 9. 结论

本次审计从 Scaling Attack 复现失败出发，发现 **5 项致命问题、5 项严重问题、10 项中等问题、5 项低级问题和 3 项信息性记录**。

核心发现可归纳为三个层面：

1. **聚合语义断裂**（F-4, F-5, S-5, M-10）：VeriFL 与标准 FedAvg 在聚合权重、范数约束、服务端优化器、BN 处理四个维度全面分歧。YAML 的 `FedAvg` 标注制造了错误的语义预期。

2. **防御框架形同虚设**（F-1, F-2, F-3, M-9）：在 VeriFL 路径下所有 FedML 原生防御被静默跳过；即使在 Baseline 路径下，Trimmed Mean 是假实现，Bulyan 不可用。17 种注册防御中仅 Krum、多 Krum 等少数经过验证的可正常工作。

3. **BN 处理碎片化**（S-1, S-3, S-4, M-6）：BN 状态在 VeriFL 的 4 个 Phase 中被反复修改（线性加权 → 不投影 → 错误应用 momentum → recalibration），处理逻辑分散在多个文件中，且 Baseline 路径完全缺少 recalibration。

这些问题不仅解释了 Scaling Attack 的失败，也预示了未来任何攻防复现实验（包括 FLTrust、FLAME 等）都将面临的结构性挑战。建议在推进新的攻防复现之前，优先完成 P0/P1 级修复。

---

> **附录：已审计文件列表**
> 
> 聚合管线：`verifl_aggregator.py`, `baseline_aggregator.py`, `server_aggregator.py`, `fedml_aggregator.py`, `agg_operator.py`, `gpu_accelerator.py`, `micro_ga_base.py`  
> 攻击框架：`fedml_attacker.py`, `label_flipping_attack.py`, `model_replacement_backdoor_attack.py`, `byzantine_attack.py`  
> 防御框架：`fedml_defender.py`, `trimmed_mean_defense.py`, `bulyan_defense.py`, `krum_defense.py`, `common/utils.py`  
> 配置传播：`run_experiment.sh`, `main_fedml_shieldfl.py`, `arguments.py`, `__init__.py`  
> 数据加载：`data_loader.py`  
> 评估管线：`asr.py`, `metrics.py`  
> 确定性控制：`runtime.py`, `fedml/__init__.py`  
> 训练管线：`verifl_trainer.py`, `client_trainer.py`, `fedml_trainer.py`  
> 服务端管理：`fedml_server_manager.py`
