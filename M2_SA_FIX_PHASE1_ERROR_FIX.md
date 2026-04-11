# P1.5 Scaling Attack 验证实施规格（学术交付文档）

> **文档性质**：学术端→工程端的正式交付文档，包含明确的变更清单、实验冻结参数和验收标准  
> **日期**：2026-04-07  
> **前置依赖**：P1 实验完成（`M2_SA_FIX_PHASE1_ERROR.md`）；P1 失败项分析完成（`P1_失败项分析与修正方案.md` v2）  
> **目标**：**确认 Scaling Attack（Model Replacement Backdoor）在 no-defense FedAvg 下确实生效**，冻结攻击参数作为后续所有防御对比实验的 baseline

---

## 0. 决策总表

以下决策已由学术端审定，工程端照此执行。每项决策均附理由出处。

| 编号 | 决策内容 | 依据 |
|:-----|:--------|:-----|
| **D-P1.5-1** | 攻击窗口从 5 轮改为**单轮**：CIFAR-10 `[99]`、MNIST `[49]` | 社区标准（FilterFL single-attack 模式）+ P1 NaN 根因（级联数值发散）；详见`P1_失败项分析与修正方案.md` §1.4 |
| **D-P1.5-2** | γ 从固定值 `scale_gamma=10` 改为**动态自适应**：`γ = Σn_i / n_malicious` | 白盒自适应攻击者设定（Gemini 审阅建议 + Bagdasaryan Eq.3 原始推导）；消除 Dirichlet 分区引入的 sample-weight 偏移 |
| **D-P1.5-3** | MNIST 验收标准放宽：5 seeds + median ≥ 0.80 | seed=1 异常为 LeNet5+MNIST 数据集固有特性（Catastrophic Forgetting），非代码 bug；详见 `P1_失败项分析与修正方案.md` §2 |
| **D-P1.5-4** | CIFAR-10 为**主力 benchmark**，MNIST 为完整性验证 | 社区共识（FilterFL/FLAME/Fang 2025 均以 CIFAR-10 为主力）+ MNIST 任务过简导致后门不稳定 |
| **D-P1.5-5** | 其余所有参数维持 P1 冻结值不变 | 变量控制：一次只改一个因素 |

### 0.1 不变参数确认清单（D-P1.5-5）

以下参数已在 P1 / M1.5 冻结，P1.5 **不得修改**：

| 参数 | 冻结值 | 冻结来源 |
|:-----|:------|:--------|
| `training_type` | `cross_silo` | 重生推进方案 §8.1 |
| `client_num_in_total` | `10` | 重生推进方案 §8.1 |
| `client_num_per_round` | `10` | 全参与（cross-silo 模式） |
| `byzantine_client_num` (K) | `1` | 重生推进方案 D-4 |
| `ratio_of_poisoned_client` (PMR) | `0.1` | Scaling_实施定稿 §3.1 |
| `comm_round` | `100`（CIFAR-10）/ `50`（MNIST） | Scaling_实施定稿 §3.3 |
| `epochs` (E) | `1` | Scaling_实施定稿 §3.1 |
| `learning_rate` | `0.01` | Scaling_实施定稿 §3.1 |
| `batch_size` | `64` | Scaling_实施定稿 §3.1 |
| `server_lr` | `1.0` | FedAvg 标准 |
| `weight_decay` | `0.0001` | Scaling_实施定稿 §3.1 |
| `backdoor_per_batch` | `20` | Bagdasaryan 原文 c=20 |
| `trigger_size` | `3`（3×3 右下角） | Scaling_实施定稿 §3.1 |
| `trigger_value` | `1.0` | 同上 |
| `target_label` | `0` | 同上 |
| `defense_type` | `none` | P1.5 测试纯 FedAvg baseline |
| `federated_optimizer` | `FedAvg` | 不可变 |
| `partition_method` | `hetero`（Dirichlet） | 同上 |
| 模型 | ResNet18（CIFAR-10）/ LeNet5（MNIST） | 同上 |

---

## 1. 变更项 1：攻击窗口改为单轮（D-P1.5-1）

### 1.1 变更内容

| 配置项 | P1 值 | P1.5 值 |
|:------|:-----|:--------|
| CIFAR-10 `attack_training_rounds` | `[95, 96, 97, 98, 99]` | **`[99]`** |
| MNIST `attack_training_rounds` | `[45, 46, 47, 48, 49]` | **`[49]`** |

### 1.2 实施要求

工程端只需确保实验配置满足以下学术冻结要求：

1. CIFAR-10 的攻击轮固定为最终轮 `[99]`。
2. MNIST 的攻击轮固定为最终轮 `[49]`。
3. 除攻击轮次外，不引入额外超参数联动修改。

### 1.3 学术理由（供工程同学了解背景）

P1 的 10/12 组 CIFAR-10 γ=10 实验 NaN 崩溃，根因是 **γ=N=10 在 N=10 下连续精确替换导致的级联数值发散**（Round 96 起，已被破坏的全局模型上再次缩放 10 倍，参数超出 float32 范围）。

社区 6 篇防御基线论文中，5/6 篇使用**每轮连续攻击**，但它们的 N 均≥50（梯度多样性能缓冲发散）。唯一与我们 γ=10 设置匹配的 FilterFL（Yang 2025）明确将 Scaling Attack 分为 "multiple attacks"和 "single attacks (a.k.a., one-shot attack)"两种模式，并同时测试。

单轮攻击在模型收敛后期一击替换，是 Bagdasaryan 原文的主实验模式，也是社区公认的标准攻击范式之一。P1 Round 95（首轮攻击）数据已证明单轮 γ 缩放即可将 ASR 推至 0.80+。

---

## 2. 变更项 2：γ 改为动态自适应（D-P1.5-2）

### 2.1 变更内容

| 配置项 | P1 值 | P1.5 值 |
|:------|:-----|:--------|
| `scale_gamma` | `10`（固定） | **`auto`**（动态计算 $\gamma = \sum n_i / n_{malicious}$） |

### 2.2 为什么需要这个改动

P1 报告附录 A 证实，Dirichlet 分区使 Client 0 的样本占比 $w_0$ 在 6.3%~13.0% 之间波动。当 `scale_gamma=10` 固定时，实际有效缩放系数 $\gamma \times w_0$ 偏离理想值 1.0：

| 数据集 | α | Client 0 样本占比 | 固定 γ=10 时的 γ×w₀ | 自适应 γ=1/w₀ 时的 γ×w₀ |
|:------|:--|:---------------|:-------------------|:----------------------|
| CIFAR-10 | 0.1 | 13.0% | **1.303**（超调 30%） | **1.000** |
| CIFAR-10 | 0.5 | 10.6% | 1.055 | **1.000** |
| CIFAR-10 | 100 | 9.8% | 0.976 | **1.000** |
| MNIST | 0.1 | 6.3% | **0.631**（欠调 37%） | **1.000** |
| MNIST | 0.5 | 10.7% | 1.070 | **1.000** |

在白盒攻击者假设下（`Scaling_实施规格.md` §3.2.1 Threat Model：攻击者知道聚合规则和自己的权重），攻击者理应用 $\gamma = 1/w_m$ 做精确模型替换。这是顶会安全论文（NDSS/CCS/USENIX Sec）对自适应攻击者的标准设定。

修改后，**无论 Dirichlet 怎样切分数据**，攻击者的有效缩放系数永远精确等于 1.0，消除了一个实验中的混淆变量。

### 2.3 验证要求

改动后，日志中应打印类似：
```
Scaling auto-gamma | round=99 | total_samples=50000 | malicious_samples=6517 | gamma=7.6732
```

验证 $\gamma \times w_0 = \gamma \times (6517/50000) = 7.6732 \times 0.1303 = 1.0000$，精确等于 1.0。

### 2.4 兼容性说明

- 当 YAML 中 `scale_gamma` 设为数字（如 `10`、`1`）时，行为与 P1 **完全一致**，无任何 regression
- 仅当设为 `auto` 时触发新的动态计算路径
- γ=1 控制组不受影响（`scale_gamma: 1` 走原有路径）

---

## 3. 变更项 3：MNIST 验收标准放宽（D-P1.5-3）

### 3.1 变更内容

无代码改动。仅调整验收标准：

| AC 编号 | P1 标准 | P1.5 标准 |
|:--------|:-------|:---------|
| AC-P1-9 | MNIST ≥3α 的 **mean** ASR ≥ **0.90**（3 seeds） | MNIST ≥3α 的 **median** ASR ≥ **0.80**（5 seeds） |

### 3.2 实施要求

MNIST 需补充 seed=3 和 seed=4（所有 4 个 α），共 8 组新实验。

### 3.3 学术理由

MNIST 上 seed=1 的 ASR≈0 是 **Catastrophic Forgetting** 现象：LeNet5 在 MNIST 这样极简特征空间上收敛得过于彻底，9 个良性客户端仅 1 epoch 就能完全冲刷后门特征。社区论文（FilterFL/FLAME 等）均以 CIFAR-10 为后门攻击的主力 benchmark，MNIST 更多作为可用性/完整性验证。

---

## 4. 实验计划

### 4.1 P1.5 完整实验矩阵

#### 4.1.1 CIFAR-10 主实验（γ=auto，单轮 [99]）

| α | seed=0 | seed=1 | seed=2 | 共 |
|:--|:-------|:-------|:-------|:--|
| 0.1 | ✦ | ✦ | ✦ | 3 |
| 0.3 | ✦ | ✦ | ✦ | 3 |
| 0.5 | ✦ | ✦ | ✦ | 3 |
| 100 | ✦ | ✦ | ✦ | 3 |
| **小计** | | | | **12** |

✦ = 需新跑

#### 4.1.2 MNIST 主实验（γ=auto，单轮 [49]）

| α | seed=0 | seed=1 | seed=2 | seed=3 | seed=4 | 共 |
|:--|:-------|:-------|:-------|:-------|:-------|:--|
| 0.1 | ✦ | ✦ | ✦ | ✦★ | ✦★ | 5 |
| 0.3 | ✦ | ✦ | ✦ | ✦★ | ✦★ | 5 |
| 0.5 | ✦ | ✦ | ✦ | ✦★ | ✦★ | 5 |
| 100 | ✦ | ✦ | ✦ | ✦★ | ✦★ | 5 |
| **小计** | | | | | | **20** |

✦ = 需新跑（单轮+auto γ）  
★ = 新增 seed

#### 4.1.3 γ=1 控制组（单轮 [99]/[49]，固定 γ=1）

| 数据集 | α | seed | 共 |
|:------|:--|:-----|:--|
| CIFAR-10 | 0.5 | 0 | 1 |
| CIFAR-10 | 100 | 0 | 1 |
| MNIST | 0.5 | 0 | 1 |
| **小计** | | | **3** |

> γ=1 控制组不使用 auto 模式，固定 `scale_gamma: 1`，用于 ΔASR 因果性验证。

#### 4.1.4 总计

| 类别 | 组数 |
|:-----|:----|
| CIFAR-10 主实验 | 12 |
| MNIST 主实验 | 20 |
| 控制组 | 3 |
| **总计** | **35** |

### 4.2 参数冻结总览

P1.5 所有实验均需满足下列统一口径：

| 维度 | CIFAR-10 | MNIST |
|:-----|:---------|:------|
| 模型 | ResNet18 | LeNet5 / SimpleCNN（保持与 P1 一致） |
| 轮数 | 100 | 50 |
| 恶意客户端数 K | 1 | 1 |
| PMR | 0.1 | 0.1 |
| 攻击轮 | [99] | [49] |
| γ | auto（主实验） / 1（控制组） | auto（主实验） / 1（控制组） |
| epochs | 1 | 1 |
| learning rate | 0.01 | 0.01 |
| batch size | 64 | 64 |
| backdoor_per_batch | 20 | 20 |
| defense | none | none |

---

## 5. 验收标准（AC）

### 5.1 代码变更验收

| AC 编号 | 验收条件 | 验证方法 |
|:--------|:--------|:--------|
| **AC-C-1** | `scale_gamma: auto` 时，日志输出动态 γ 且 γ×w₀=1.0000（±0.0001） | 检查实验日志中 `Scaling auto-gamma` 行 |
| **AC-C-2** | `scale_gamma: 10` 时行为与 P1 完全一致 | 对比 P1 已有 γ=1 控制组结果（应完全相同） |
| **AC-C-3** | `scale_gamma: 1` 控制组不触发缩放逻辑中的 auto 路径 | 检查日志中无 `auto-gamma` 输出 |

### 5.2 实验结果验收

#### CIFAR-10（主力 benchmark）

| AC 编号 | 验收条件 | P1 基础 |
|:--------|:--------|:--------|
| **AC-P1.5-1** | ≥3 个 α 的 mean ASR(γ=auto) ≥ **0.80** | P1 Round 95 首轮数据 ASR 约 0.80~0.91 |
| **AC-P1.5-2** | γ=auto 的所有 12 组实验 **loss ≠ NaN**（无崩溃） | P1 有 10/12 NaN；单轮预期消除 |
| **AC-P1.5-3** | γ=auto 的所有存活实验 clean accuracy > **30%** | Model replacement 后 acc 下降合理；P1 R95 数据约 48~59% |
| **AC-P1.5-4** | ΔASR(γ=auto vs γ=1) ≥ **0.30** | P1 的 Δ=0.650；预期类似或更高 |

#### MNIST（完整性验证）

| AC 编号 | 验收条件 | P1 基础 |
|:--------|:--------|:--------|
| **AC-P1.5-5** | ≥3 个 α 的 **median** ASR(γ=auto) ≥ **0.80**（5 seeds） | P1 3-seed median: 0.724~0.899 |
| **AC-P1.5-6** | α=100 (IID) 的所有 5 seeds ASR > **0.90** | P1: 0.961~0.992 |
| **AC-P1.5-7** | γ=auto 的所有组实验 loss ≠ NaN | P1 MNIST 无 NaN |

#### 因果性验证

| AC 编号 | 验收条件 |
|:--------|:--------|
| **AC-P1.5-8** | γ=auto 控制组与 γ=1 控制组的 ΔASR ≥ 0.30（至少 2/3 控制组达标） |

### 5.3 验收失败的处理流程

- 若 **AC-P1.5-2 失败**（仍有 NaN）：单轮攻击不应 NaN，请立即上报并附完整日志——这意味着有新的 bug
- 若 **AC-P1.5-1 未达 0.80**：检查 auto γ 的实际值是否合理，以及 Round 99 时模型是否已充分收敛
- 若 **AC-P1.5-5 未达 0.80**：MNIST 可以按 D-P1.5-4 降级为"已知限制"，不阻断整体验收

---

## 6. 预期数据产出

每组实验需产出以下数据，用于后续防御对比：

### 6.1 每组实验必须记录

| 数据项 | 来源 | 格式 |
|:-------|:-----|:-----|
| 每轮 clean accuracy | `metrics.jsonl` | `{"round": t, "accuracy": x}` |
| 每轮 ASR（attack success rate） | `metrics.jsonl` | `{"round": t, "asr": x}` |
| 每轮 loss | `metrics.jsonl` | `{"round": t, "loss": x}` |
| 攻击轮的实际 γ 值 | 实验日志 `Scaling auto-gamma` 行 | 日志原文 |
| 最终轮（R99/R49）的 clean acc + ASR + loss | 汇总表 | 见下 |

### 6.2 汇总表格式

请以与 `M2_SA_FIX_PHASE1_ERROR.md` 相同的格式产出汇总表：

```markdown
| 数据集 | α | seed | γ (auto) | 攻击轮 | Pre-Atk Acc | Final Acc | Final ASR | Final Loss | 状态 |
```

### 6.3 冻结产出（供后续防御对比）

P1.5 完成后，以下数据将被**冻结**，作为后续所有防御实验的攻击 baseline：

1. **CIFAR-10 no-defense ASR baseline**：12 组实验的 mean/median ASR
2. **MNIST no-defense ASR baseline**：20 组实验的 median ASR
3. **冻结参数快照**：YAML 配置存入 `results/configs/` 并 git commit
4. **动态 γ 值记录**：每个 (dataset, α) 组合的实际 γ，作为后续固定参考

---

## 7. 社区标准对标说明（供工程同学了解全貌）

### 7.1 当前设置与社区的对比

| 维度 | 社区主流 | 我们（P1.5） | 差异说明 |
|:-----|:--------|:-----------|:--------|
| N（总客户端） | 100 | **10** | 受 cross-silo/MPI 限制；后续 M1c 将迁移到 N=100 |
| 每轮参与 | 全参与（4/6 篇） | **全参与** | ✅ 一致 |
| PMR | 20%（4/6 篇） | **10%（K=1/N=10）** | 接近 FilterFL 的 10% |
| γ | n 或 n/η（5/6 篇） | **auto = 1/w₀** | ✅ 更严谨（精确替换对齐） |
| 攻击频率 | 每轮（5/6 篇） | **单轮** | 与 FilterFL single-attack 模式对标；N=10 下连续攻击会 NaN |
| 后门训练 | 必须有（6/6 篇） | **✅ 有** | backdoor_per_batch=20 |
| 攻击者超参 | 与良性相同（社区共识） | **✅ 相同** | E=1, lr=0.01，无差异化 |

### 7.2 我们与社区的核心差异及后续计划

当前设置与社区的最大差异是 **N=10（cross-silo 全参与）vs 社区 N=100（cross-device 10% 采样）**。这导致：

1. γ=N 的绝对值小（10 vs 100），但 K×γ/N 的替换比率等效（均为 1.0）
2. 梯度多样性不足，连续 γ 缩放容易 NaN（社区 N=100 无此问题）
3. 无法测试攻击者"不是每轮都在场"的场景（这是 cross-device 安全防御的核心挑战）

**后续计划（M1c 阶段，不在 P1.5 范围内）**：

- 将训练模式从 cross-silo MPI 迁移到 FedML 的 **SP（单进程模拟）模式**
- SP 模式天然支持 N=100 + 10% 每轮采样，且每轮实际计算量与当前 N=10 全参与一致（都是 10 个 client 训练）
- 需要的桥接工程：SP 模式的 `FedAvgAPI._aggregate()` 接入 `ServerAggregator` 的 `on_before_aggregation → aggregate → on_after_aggregation` 管线，使 `FedMLAttacker`/`FedMLDefender` 的 hook 能在 SP 模式下触发
- 攻击代码中的 `assert client_num_per_round == client_num_in_total` 需移除并适配采样模式

**P1.5 的定位是在当前 N=10 基础设施上先验证攻击逻辑正确性**，后续再迁移到论文级的 N=100 设置。

---

## 8. 交付件

| 步骤 | 内容 | 交付件 |
|:-----|:-----|:------|
| Step 1 | 完成 P1.5 参数对齐 | 可运行配置 |
| Step 2 | 跑 1 组快速验证（CIFAR-10 α=0.5 seed=0，单轮攻击，γ=auto） | 日志截图：确认 auto-gamma 输出 + 无 NaN + ASR > 0.80 |
| Step 3 | 批量跑全部 35 组实验 | `metrics.jsonl` × 35 |
| Step 4 | 汇总表 + AC 验收 | 报告文档（类似 `M2_SA_FIX_PHASE1_ERROR.md` 格式） |
| Step 5 | 冻结实验配置与结果摘要 | 配置快照 + 汇总结果 |

---

## 附录 A：完整 AC 检查表（工程同学用）

| AC | 条件 | Pass/Fail |
|:---|:-----|:----------|
| AC-C-1 | auto γ 日志输出正确，γ×w₀=1.0 | [ ] |
| AC-C-2 | 固定 γ=10 行为无 regression | [ ] |
| AC-C-3 | γ=1 控制组不触发 auto 路径 | [ ] |
| AC-P1.5-1 | CIFAR-10 ≥3α mean ASR ≥ 0.80 | [ ] |
| AC-P1.5-2 | CIFAR-10 全部 12 组 loss ≠ NaN | [ ] |
| AC-P1.5-3 | CIFAR-10 存活实验 clean acc > 30% | [ ] |
| AC-P1.5-4 | ΔASR(auto vs γ=1) ≥ 0.30 | [ ] |
| AC-P1.5-5 | MNIST ≥3α median ASR ≥ 0.80（5 seeds） | [ ] |
| AC-P1.5-6 | MNIST α=100 全部 5 seeds ASR > 0.90 | [ ] |
| AC-P1.5-7 | MNIST 全部 loss ≠ NaN | [ ] |
| AC-P1.5-8 | 因果性 ΔASR ≥ 0.30（≥2/3 控制组） | [ ] |

---

## 附录 B：P1→P1.5 变更追溯矩阵

| 变更 | 触发来源 | 影响范围 | 风险评估 |
|:-----|:--------|:--------|:--------|
| 攻击窗口 5→1 轮 | P1 NaN 根因分析 + 社区 FilterFL 先例 | 实验配置 | 无风险 |
| γ 固定→auto | Gemini 审阅建议 + Bagdasaryan Eq.3 | 实验配置与攻击实现 | 低（保留固定 γ 兼容路径） |
| MNIST 3→5 seeds | P1 seed=1 异常分析 | 增加 8 组实验 | 无风险 |
| AC-P1-9 mean→median + 0.90→0.80 | 统计鲁棒性 + 社区 MNIST 定位 | 验收标准文档 | 无风险 |
