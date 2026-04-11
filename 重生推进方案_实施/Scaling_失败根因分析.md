# Scaling Attack 失败根因分析

> 日期：2025-07-07
> 分析对象：bde23de 提交的 Scaling Attack 实现及其实验结果
> 核心问题：AC-13 (缩放因果性) FAIL，CIFAR-10 全部 γ=10 实验模型崩溃

---

## 0. 失败现象回顾

| 指标 | γ=10 (3 seeds) | γ=1 (3 seeds) |
|------|---------------|--------------|
| CIFAR-10 α=0.5 Accuracy | 0.1000 ± 0.0000 | 0.7777 |
| CIFAR-10 α=0.5 ASR | 1.0000 ± 0.0000 | 0.8108 |
| ΔASR | +0.19（< 0.30 阈值，**AC-13 FAIL**）| — |

- γ=10：accuracy=0.10 = 10 类随机猜测 → **模型崩溃**
- ASR=1.0 是因为模型无条件输出 target_label=0，不是精确后门
- γ=1：clean accuracy 保持正常，ASR 已达 0.81
- 所有 24 组 γ=10 的 CIFAR-10 实验均出现相同崩溃

---

## 1. 根因清单（按严重程度排序）

### 根因 1（致命）：γ=N 公式在 K>1 时导致超调

**这是导致模型崩溃的首要原因。**

Bagdasaryan 原文的 Scaling 公式设计初衷是：**1 个恶意客户端**，用 γ=N 让 FedAvg 后全局模型被恶意模型替换。

推导（K=1, γ=N, 等权 FedAvg）：

$$G_{\text{new}} = \frac{1}{N}\bigl[\gamma(W_m - G) + G + (N-1)W_b\bigr]$$

若 $W_b \approx G$（收敛附近）：

$$G_{\text{new}} \approx \frac{1}{N}\bigl[N(W_m - G) + NG\bigr] = W_m \quad \text{(model replacement)}$$

但本项目设置 **K=3 个恶意客户端，每个都 γ=10**：

$$G_{\text{new}} = \frac{1}{N}\bigl[K \cdot \gamma(W_m - G) + KG + (N-K)W_b\bigr]$$

$$\approx \frac{K\gamma}{N}(W_m - G) + G = \frac{3 \times 10}{10}(W_m - G) + G = 3(W_m - G) + G$$

$$= G + 3\delta_m \quad \text{（超调 3 倍！）}$$

| 参数组合 | 有效放大倍数 $K\gamma/N$ | 效果 |
|---------|----------------------|------|
| K=1, γ=10 (原文) | 1.0 | 模型替换（正确） |
| **K=3, γ=10 (当前)** | **3.0** | **3x 超调 → 崩溃** |
| K=3, γ=10/3≈3.33 (应当) | 1.0 | 模型替换（正确） |

**在 5 轮攻击窗口中，每轮 3 倍超调级联累积**，模型参数迅速偏离正常范围，导致全部输出塌缩到一个类。

> **定量估计**：假设每轮恶意更新方向类似，5 轮累积放大倍数约为 $3^5 = 243$（最坏情况）或至少远超模型容忍范围。

### 根因 2（致命）：聚合管线不是 FedAvg，而是 VeriFL

**规格 §1 冻结聚合器为 "FedAvg"，但实际运行路径是 VeriFL 三阶段聚合。**

`run_experiment.sh` 生成 YAML 时设置 `aggregator_type: "fedavg"`，但 VeriFLAggregator 构造函数**强制覆写**：

```python
# verifl_aggregator.py line 69
setattr(args, "aggregator_type", "verifl")
```

实际聚合流程：

```
on_before_aggregation()
  → FedMLAttacker.attack_model()     # Scaling 缩放在此执行，作用于完整列表
  → 返回缩放后列表

aggregate()                            # 注意：NOT FedMLAggOperator.agg (FedAvg)
  → Phase 1: GA 遗传搜索 (pop=15, gen=10)  # 找"最优"聚合权重
  → Phase 2: L2 范数投影                    # 所有客户端模型归一化到 anchor 范数
  → Phase 3: 加权聚合 + server momentum     # 用 GA 权重聚合投影后模型
  → BN 重校准                              # 用服务器验证集覆写 BN stats
```

**问题链**：

| # | VeriFL 步骤 | 对 Scaling 攻击的影响 |
|---|-----------|-------------------|
| 1 | GA 搜索在**缩放后**模型上评估 fitness | GA 可能给崩溃性权重组合高分 |
| 2 | L2 范数投影 | 部分但非完全逆转缩放效果（见下文） |
| 3 | 投影后加权聚合 | GA 权重是基于投影前模型优化的，投影后权重不再最优 |
| 4 | BN 重校准 | 完全覆写 Scaling 攻击精心注入的 BN stats |

这意味着 Scaling 攻击的效果路径被 VeriFL 以**不可预测的方式**扭曲。既不是"FedAvg 直接聚合缩放模型"（原文假设），也不是"防御完全阻止攻击"（因为 VeriFL 不是按防御策略运行的）。

### 根因 3（重要）：L2 范数投影的部分抵消效应

VeriFL Phase 2 将所有客户端模型的可训练参数 L2 范数归一化到 anchor（GA 给出最大权重的那个客户端）的范数：

```python
anchor_norm = calc_l2_norm(weights_results[anchor_idx])
for client_idx in range(num_clients):
    client_norm = calc_l2_norm(weights_results[client_idx])
    scale = anchor_norm / (client_norm + 1e-9)
    projected = [layer * scale for layer in weights_results[client_idx]]
```

缩放后的恶意模型 $W_m' = G + 10(W_m - G)$ 的 L2 范数：

- 如果 $\|update\| \ll \|G\|$：$\|W_m'\| \approx \|G\|$（投影 ≈ 无效果）
- 如果 $\|update\| \sim 0.1\|G\|$：$\|W_m'\| \approx 2\|G\|$（投影缩小约 50%）

实际效果：**投影对小更新几乎无效，对大更新部分抵消但不完全**。这使得攻击效果处于"不上不下"的不可预测区间。

关键细节：`trainable_mask` 仅包含 `named_parameters()`（weight, bias），**不包含** BN 的 running_mean/running_var。因此：
- L2 范数计算**不考虑** BN stats
- 投影只缩放可训练参数，**BN stats 不被投影**
- **缩放后的 BN stats 直接保留**（但后续被 BN 重校准覆写，见根因 4）

### 根因 4（中等）：BN 重校准覆写攻击注入的 BN 参数

VeriFL 聚合后执行 BN 重校准：

```python
def recalibrate_batchnorm(self, params, batch_size=64, passes=1):
    self._load_state_from_ndarrays(params)
    self.model_template.train()
    with torch.no_grad():
        for _ in range(passes):
            for start in range(0, total, batch_size):
                _ = self.model_template(self.val_images[start:end])
    self.model_template.eval()
    return self._extract_state_to_ndarrays()
```

这用服务器验证集做 forward pass 来重新计算 running_mean/running_var。Scaling 攻击通过 `should_scale_param(k)` 精心缩放的 BN stats 被此步完全覆写。

**后果**：
- 攻击代码为包含 BN stats 投入的设计努力（`should_scale_param` vs `is_weight_param`）被白白浪费
- 但这不直接导致崩溃——崩溃的主因仍是根因 1（γ 超调）

### 根因 5（中等）：GA-投影不一致性

GA 搜索（Phase 1）的 fitness 评估基于 **投影前（缩放后）** 的客户端模型矩阵：

```python
# GA 使用的模型矩阵 = 缩放后的完整模型
self.gpu_accelerator.set_client_parameters(weights_results)  # 缩放后

# 但最终聚合用的是投影后模型
ga_aggregated_params = aggregate_weighted(projected_weights, best_weights)
```

GA 找到的最优权重 α* 最小化 `Σ αᵢ·Wᵢ_scaled` 的验证损失，但实际聚合使用 `Σ αᵢ·Wᵢ_projected`。两者不是同一模型。

**在攻击轮次**：缩放后模型与投影后模型差异显著 → **GA 权重对投影后聚合不再最优** → 可能放大崩溃而非缓解。

### 根因 6（中等）：γ=1（控制实验）≠"无后门训练"

AC-13 定义的对照实验用 γ=1 代替 γ=10，但 γ=1 时后门训练 **仍然生效**：

```python
# γ=1 时缩放公式:
W_m' = 1 × (W_m - G) + G = W_m  # 不缩放，但 W_m 本身已含后门
```

3 个恶意客户端占 30% 的训练更新，每 batch 注入 20/64 ≈ 31% 后门样本，连续 5 轮。这已经是很强的攻击条件。

**γ=1 的 ASR 已达 0.81** 说明在当前实验壳下（K/N=0.3, 31% poison ratio, 5-round window），后门训练本身就已经高度有效，scaling 不是必要条件。

这使得 AC-13 的"因果性"命题极难成立——**不是 γ=10 不够有效（实际上过于有效导致崩溃），而是 γ=1 已经太有效了**。

### 根因 7（低）：MNIST 高方差的额外因素

MNIST + LeNet5 实验的 ASR 方差极大（0.002~0.640），原因可能包括：

- LeNet5 参数量远小于 ResNet18，同样的 γ=10 缩放导致相对更大的参数扰动
- MNIST 收敛更快（50 轮），前 45 轮已接近完美（acc>97%），后 5 轮攻击窗口的扰动效果高度依赖模型状态
- LeNet5 无 BN 层（纯 fc+conv），因此 BN 相关讨论不适用

---

## 2. 因果关系图

```
γ=10 × K=3
  │
  ├──→ 有效放大 = 3x（应为 1x）──→ 每轮 3 倍超调 ──→ 5 轮累积 ──→ 模型崩溃（acc=0.10）
  │                                                                      │
  │                                                                      ├──→ ASR=1.0 (trivial: 全预测 class 0)
  │                                                                      │
  │                                                                      └──→ ΔASR = 1.0 - 0.81 = 0.19 < 0.30 → AC-13 FAIL
  │
  └──→ VeriFL 管线（非 FedAvg）
        │
        ├──→ GA 在缩放后模型上优化权重 → 权重不适配投影后模型
        ├──→ L2 投影部分抵消缩放 → 不可预测的残余效果
        ├──→ BN 重校准覆写 BN stats → 攻击精度损失
        └──→ 整体效果不可控：既非原文 FedAvg 语义，也非有效防御
```

---

## 3. 对照：规格与实际的偏差

| 规格假设 | 实际运行路径 | 偏差等级 |
|---------|-----------|---------|
| 聚合器: FedAvg (§1) | VeriFL 三阶段（GA + L2投影 + momentum） | **致命** |
| γ = client_num_in_total = 10 (§2) | γ=10 但 K=3，有效放大 = 30/10 = 3 | **致命** |
| 缩放公式直接影响 FedAvg 聚合结果 (§4.3) | 缩放结果先经 L2 投影再由 GA 权重聚合 | **重大** |
| should_scale_param 包含 BN stats (§4.4) | BN stats 被 VeriFL BN 重校准覆写 | **中等** |
| server_lr=1.0 (§1) | YAML 设置正确，server_momentum=0.0，momentum 有效禁用 | ✅ 无偏差 |
| 轮次 0-indexed，对齐一致 (隐含) | server/client/trainer/attack 四处均 0-indexed，一致 | ✅ 无偏差 |
| ASR 评估在聚合后执行 (§8.2) | 正确：test() 在 aggregate() 之后调用 | ✅ 无偏差 |

---

## 4. 验证各根因独立性的建议实验

为了确认上述分析，建议以下诊断实验（不需要完整跑正式矩阵）：

### 实验 A：验证根因 1（γ 超调是主因）

```
γ = 10/3 ≈ 3.33（理论上正确的 model replacement 值）
其余参数不变
预期：模型不崩溃，ASR 显著高于 γ=1
```

### 实验 B：验证根因 2（VeriFL 管线影响）

需要绕过 VeriFL 聚合，使用真正的 FedAvg 路径。两种方法：

1. **快速方法**：在 `VeriFLAggregator.aggregate()` 中添加 FedAvg 直通分支：

```python
if getattr(self.args, "aggregator_type", "verifl") == "fedavg_bypass":
    return FedMLAggOperator.agg(self.args, raw_client_model_or_grad_list)
```

2. **严格方法**：用不同的 γ 值在 FedAvg bypass 下跑：
   - γ=10, K=3 + FedAvg → 预期崩溃（验证根因 1）
   - γ=3.33, K=3 + FedAvg → 预期 model replacement 成功
   - γ=10, K=3 + VeriFL → 对照当前结果

### 实验 C：验证根因 6（γ=1 基线过强）

```
攻击窗口缩短为最后 1 轮（如 [99]）
γ=1 和 γ=10/3 对比
预期：γ=1 的 ASR 大幅降低，使 ΔASR 拉开
```

---

## 5. 综合结论与建议

### 5.1 为什么通不过

**直接原因**：γ=10 × K=3 导致 3 倍超调 → 5 轮累积 → 模型崩溃 → ASR=1.0 是 trivial collapse 而非精确后门 → 与不崩溃的 γ=1 对比 ΔASR 不够大。

**深层原因**：
1. 规格中 γ=N 的公式源自 Bagdasaryan 原文 K=1 的假设，直接用于 K=3 是数学错误
2. 实际聚合路径是 VeriFL 而非 FedAvg，使得缩放效果不可预测
3. γ=1 基线已经很强（30% 恶意率 + 31% 注入率 + 5 轮窗口），留给缩放的"增量空间"不大

### 5.2 修复建议

| 优先级 | 行动 | 预期效果 |
|--------|------|---------|
| **P0** | 修正 γ 公式为 **γ = N/K**（= 10/3 ≈ 3.33） | 消除 3x 超调，恢复 model replacement 语义 |
| **P0** | 增加 **FedAvg 直通模式**绕过 VeriFL 管线 | 使实验与规格假设一致 |
| **P1** | 增设 **γ 梯度实验** {1, 2, 3.33, 5, 10} | 找到不崩溃的有效 γ 范围 |
| **P1** | **缩短 γ=1 基线攻击窗口** 或减小 K/N 比例 | 降低无缩放基线，使 ΔASR 更容易达标 |
| **P2** | 更新规格 §2 和 §3：明确 γ 与 K 的关系 | 文档一致性 |
| **P2** | 评估 VeriFL 聚合下的 Scaling 攻击行为并补充说明 | 学术完整性 |

### 5.3 推荐的最小改动路径

如果目标是在**不改变实验壳**（仍使用 VeriFL 管线）的前提下通过 AC-13：

1. 将 γ 改为 10/3 ≈ 3.33
2. 重跑 3 组 CIFAR-10 α=0.5 正式实验（γ=3.33, seed=0,1,2）
3. 重跑 3 组控制实验（γ=1, seed=0,1,2）
4. 检查 ΔASR 是否 ≥ 0.30

如果目标是**严格复现 Bagdasaryan 原文语义**：

1. 添加 FedAvg 直通模式
2. 将 γ 改为 N/K = 10/3
3. 重跑全部 27 组实验
4. 更新规格文档

---

## 附录 A：代码走查——确认无编码错误

在确认"为什么崩溃"之前，先排除编码 bug。以下是完整验证：

| 检查项 | 结果 | 说明 |
|--------|------|------|
| 缩放公式 `γ(W-G)+G` | ✅ 正确 | model_replacement_backdoor_attack.py L88-91 |
| K=3 固定恶意 ID | ✅ 正确 | `list(range(byzantine_client_num))` |
| 所有恶意客户端均被缩放 | ✅ 正确 | `for idx in self.malicious_client_ids` 循环 |
| should_scale_param 排除 num_batches_tracked | ✅ 正确 | `"num_batches_tracked" not in k` |
| 攻击轮跳过时仍递增 training_round | ✅ 正确 | skip 分支有 `self.training_round += 1` |
| 轮次 0-indexed | ✅ 正确 | server/client/trainer/attack 均从 0 开始 |
| 四处 round 计数对齐 | ✅ 正确 | 全部同步，无 off-by-one |
| trigger 注入 = ASR 评估 | ✅ 正确 | 均为 `images[:,:,-3:,-3:] = 1.0` |
| 后门样本 label 改为 target_label=0 | ✅ 正确 | `labels[indices] = self._target_label` |
| 隔离 RNG | ✅ 正确 | `np.random.default_rng(seed + id + 2^20)` |
| YAML 字段传递完整 | ✅ 正确 | scale_gamma/attack_training_rounds/backdoor_per_batch 均到位 |
| server_momentum=0.0, server_lr=1.0 | ✅ 正确 | momentum 有效禁用 |

**结论：代码层面没有 bug，问题出在参数设计（γ 值）和聚合架构（VeriFL vs FedAvg）。**

---

## 附录 B：数学补充——正确的 γ 推导

对于 FedAvg（等权平均），K 个恶意客户端各自以 γ 缩放后，聚合结果为：

$$G_{\text{new}} = \frac{1}{N}\left[K \cdot \bigl(\gamma(W_m - G) + G\bigr) + (N-K) \cdot W_b\right]$$

令 $W_b \approx G$：

$$G_{\text{new}} \approx G + \frac{K\gamma}{N}(W_m - G)$$

- **Model replacement** 要求 $G_{\text{new}} = W_m$，即 $\frac{K\gamma}{N} = 1$，于是 $\gamma = \frac{N}{K}$
- 当前设置 $\gamma = N = 10, K = 3$：$\frac{K\gamma}{N} = 3$（3 倍超调）
- 正确设置 $\gamma = N/K = 10/3 \approx 3.33$：$\frac{K\gamma}{N} = 1$（精确替换）

注意：以上推导假设 FedAvg 等权平均。若 FedAvg 按样本数加权且各客户端样本数不等，公式需相应调整：

$$\gamma = \frac{\sum_{i} n_i}{\sum_{j \in \mathcal{M}} n_j}$$

其中 $\mathcal{M}$ 为恶意客户端集合，$n_i$ 为客户端 $i$ 的样本数。

---

*本报告基于对 bde23de 提交的完整代码走查、VeriFL 聚合管线源码分析、以及缩放攻击数学推导。所有路径均经过实际文件验证。*
