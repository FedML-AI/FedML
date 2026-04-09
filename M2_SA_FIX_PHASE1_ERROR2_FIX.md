# Scaling Attack 修复与验证方案（v2.1 — 学术决策终稿）

> **文档性质**：学术端正式交付文档，整合 P1/P1.5 全部教训与质询交流结论，定义进入 N=50 正式实验前的所有学术决策、参数规格和验收标准  
> **日期**：2026-04-09 (v2.1 更新：PMR 精简、comm_round 调整、E=1 决策形式化)  
> **最后修订**：2026-04-08 (v2.1 审计补充：D-12 Dirichlet cap、D-13 JSONL 增补、D-14 ASR 指标定义、γ 社区对比、Phase 0 扩展、风险登记更新)  
> **前置输入**：  
> - `P1.5_AC失败根因分析.md`（两大根因 RC-1/RC-2 定位）  
> - `P1.5_修复验证方案.md`（v1 方案，含 Gemini 事实核查）  
> - `P1.5修复验证方案读后的质询与交流.md`（两轮质询 + 三项战略决策）  
> - `M2_SA_FIX_PHASE1_ERROR2.md`（P1.5 实验数据 35 组）  
> - `M2_SA_FIX_PHASE1_ERROR_FIX.md`（P1.5 实施规格 D-P1.5-1~5）  
> - `docs/M2_research/Scaling复现/Scaling_参数审计_六论文对比.md`（6 篇论文全景对照）  
> - `cross_path_strategy/04_最终学术执行方案_冻结版.md`（N=50 冻结决策）  
> - `阶段性复盘_实验平台审计报告.md`（平台能力审计）  
> - `M1_EXPERIMENT_PARAMS.md`（M1 冻结参数）  
> - 6 篇 References 原文 + FedSecurity (KDD'24) 基准论文  
> **状态**：**FROZEN** — 决策不再回退，直接进入工程实施与实验执行

---

## 〇、文档定位与废弃声明

本文档 **取代** 以下旧文档中涉及 Scaling Attack 修复的部分：

| 旧文档 | 废弃范围 | 原因 |
|:-------|:--------|:-----|
| `P1.5_修复验证方案.md` | **全部废弃** | 修复项 F-1（BN 排除缩放）经二次独立分析后推翻；F-2 方案 A（配置层修改）被方案 B 取代；实验规模从 N=10 升级为 N=50 |
| `M2_SA_FIX_PHASE1_ERROR_FIX.md` | §0 决策表中 D-P1.5-1（单轮攻击）、D-P1.5-5（冻结 N=10）废弃 | 切换为 N=50 持续攻击 |
| `cross_path_strategy/04_最终学术执行方案_冻结版.md` | §1.2（One-shot 攻击策略）废弃 | 改为持续攻击（社区标准） |
| `Scaling_实施定稿.md` | §3.1 中 `client_num_in_total: 10` 废弃 | 升级为 50 |

本文档中的决策如与上述旧文档冲突，**以本文档为准**。

---

## 一、决策总表

下表汇总本方案的全部学术决策。每项决策均附社区出处和推导过程。

| 编号 | 决策内容 | 社区依据 | 与 v1 方案（P1.5 修复验证方案）的差异 |
|:-----|:--------|:--------|:----------------------------------|
| **D-1** | 实验规模直接升级为 **N=50, m=50（全参与）**，跳过 N=10 验证 | FLAD (TDSC'25) N=50 全参与；FLTrust/FLAME/Fang/FLDetector N=100 全参与 | v1 建议先 N=10 验证再切 N=50 → **废弃** |
| **D-2** | 攻击频率改为 **每轮持续攻击** | 5/6 篇参考论文使用每轮攻击（FLTrust/FLAD/FLAME/FLDetector/Fang） | v1 沿用 P1.5 单轮攻击 → **废弃** |
| **D-3** | BN 统计量 **保持全参数缩放**，不做任何排除处理 | 6/6 篇论文描述 "scale entire model update"，无一区分 BN buffers vs learnable params | v1 建议 F-1 排除 BN → **废弃** |
| **D-4** | Trigger value 采用 **方案 B（代码层自适应转换）**：config 中 `trigger_value` 语义为原始像素空间值，代码自动转为归一化空间 | 6/6 篇论文的 trigger 均在像素空间定义 | v1 建议 F-2 方案 A（配置层手动算）→ **升级为方案 B** |
| **D-5** | auto-γ **保持现有实现**（γ = total_samples / malicious_samples），不加 gamma_cap | 白盒攻击者标准设定；N=50 下 γ ≈ 5–10，无 NaN 风险 | v1 根因分析中提出 gamma_cap → **不再需要** |
| **D-6** | PMR 测试集 **仅 20%** | FLTrust/FLAD/Fang 默认 20%；4/6 篇以 20% 为主表实验 | **v2 的 {10%,20%} 精简为仅 20%** |
| **D-7** | 攻击者本地超参 **与良性客户端一致**（E=1, lr=0.01） | FLAD 明确 "as honest client"；其余论文未单独指定 | 与 v1 一致 |
| **D-8** | weight_decay：CIFAR-10 = 0.0001，MNIST = 0.0 | M1 冻结参数配方；ResNet 社区标准 vs LeNet5 无需 | 与 M1 一致 |
| **D-9** | 保留 MNIST 测试 | 6/6 篇论文均测 MNIST 后门攻击 | 与 v1 一致 |
| **D-10** | JSONL 写入幂等化 | 数据管理审慎性 | 与 v1 一致 |
| **D-11** | 训练制度采用 **E=1 (epoch-based)**，不采用 $R_l=1$ (single-step) | McMahan 2017 FedAvg 原文 E∈{1,5,20}；FLAD (TDSC'25) Algorithm 1 epoch-based；FLTrust $R_l=1$ 仅为其 Theorem 1 理论要求，非通用标准 | **v2.1 新增**，形式化既有配置选择 |
| **D-12** | Dirichlet 分区 **移除 FedML 原有 Boolean cap**，改用纯 Dirichlet + 最小样本保护 | 6/6 篇社区论文使用无 cap 的标准 Dirichlet；cap 削弱 α=0.1 下的实际异质性，导致论文声称的 non-IID 强度失真；移除 cap 为一劳永逸的平台对齐 | **v2.1 新增**，审计后决策升级为移除 |
| **D-13** | JSONL 结构化日志 **增补 gamma/ASR/攻击元信息字段** | AC-C-1 自动化验证需要结构化 gamma 数据；社区实验结果可追溯性标准 | **v2.1 新增**，审计中发现的可观测性缺口 |
| **D-14** | ASR 指标定义为 **FinalASR（末轮值）**，同时记录 MaxASR（历史峰值） | FLTrust/FLAD/Fang/FedSecurity 均报告 final round ASR；MaxASR 作为补充指标预防审稿人追问 | **v2.1 新增**，指标定义形式化 |

---

## 二、决策详细论证

### 2.1 D-1：N=50 全参与，跳过 N=10

#### 2.1.1 为什么不再在 N=10 上做验证

1. **N=10 数据不可用于论文**：6 篇参考论文中 N 最小为 50（FLAD），N=10 无直接可比对象。审稿人必然质疑规模过小、防御方不现实地有利（每轮可见全部客户端且仅 1 个攻击者）。
2. **代码逻辑已充分验证**：P1（51 组）+ P1.5（35 组）共 86 组实验，所有失败均归因于 **配置/参数层**（γ 值、trigger 空间、BN 临界条件），不存在代码 bug。三个 AC-C-* 全部 PASS 已确认核心攻防管线正确。
3. **计算成本不增加**：N=50 时每客户端 CIFAR-10 数据量 = 50000/50 = 1000（vs N=10 时 5000），每客户端每 epoch batch 数 = 1000/64 ≈ 16（vs 78）。总 passes = 50×T×16 ≈ N=10 时 10×T×78 的同一量级。

#### 2.1.2 N=50 的社区对标

| 参数 | FLAD (TDSC'25) | 本方案 |
|:-----|:-------------|:------|
| N 总客户端 | 50 | **50** ✓ |
| m 每轮参与 | 50（全参与） | **50** ✓ |
| 数据集 | MNIST, CIFAR-10 | **MNIST, CIFAR-10** ✓ |
| 模型 | CNN(MNIST), ResNet-18(CIFAR-10) | **LeNet5(MNIST), ResNet-18(CIFAR-10)** |
| 通信后端 | — | MPI (cross-silo) |

> **模型差异说明**：FLAD 的 MNIST CNN（2conv+1FC, 5×5 kernel, 10/20 channels）与我们的 LeNet5（2conv+3FC, 5×5 kernel, 6/16 channels）均为轻量级 CNN，参数量同一量级，差异不影响攻防结论的可比性。社区对 MNIST 模型本身无严格统一标准。

#### 2.1.3 显存可行性验证

基于 `cross_path_strategy/04_最终学术执行方案_冻结版.md` §2.1 精确计算：

- ResNet18 单进程显存 ≈ 657 MB（模型 43MB + 梯度 43MB + 优化器 43MB + 激活值 128MB + CUDA 400MB）
- N=50 + 1 server = 51 进程，总需 ≈ 33 GB
- 2 张 4090（24GB × 2 = 48GB）即可满足

**前置验证要求**：正式跑 N=50 前，先在单张 4090 上拉 13 个 worker 跑 3 轮 smoke test，用 `nvidia-smi` 确认实际峰值显存。

---

### 2.2 D-2：每轮持续攻击

#### 2.2.1 社区攻击频率全景

| 论文 | 攻击频率 | 原文证据 |
|:-----|:--------|:--------|
| FLTrust (NDSS'21) | **每轮** | "in each iteration of FL, each malicious client computes..." |
| FLAME (USENIX Sec'22) | **每轮** | 全文无 "single-shot" 限定，攻击持续注入 |
| FLDetector (S&P'22) | **每轮** | "the malicious clients perform attacks **in every iteration** of FL training" |
| Fang 2025 (TDSC) | **每轮** | "every round" |
| FLAD (TDSC'25) | **每轮** | "attacks in every round" |
| FilterFL (CCS'25) | **末段 10% 轮次** | M 型每轮；S 型仅最后 10% 且 scale ×10 |

**结论**：5/6 篇论文以每轮连续攻击为默认配置。**持续攻击是社区标准**。

#### 2.2.2 为什么 N=10 时不能持续攻击但 N=50 可以

P1 的 NaN 崩溃根因是 **N=10 小规模下连续精确替换的级联数值发散**（详见 `P1_失败项分析与修正方案.md` §1.2.2）：

- N=10 时仅 9 个良性客户端，梯度多样性极低，连续替换后良性恢复力不足
- Round 96 起，全局模型已被替换，各客户端从被损坏的基础训练再被缩放，参数超出 float32 → NaN

N=50 时情况根本性改变：

| 维度 | N=10 + 单轮 | N=50 + 每轮 |
|:-----|:-----------|:-----------|
| 恶意客户端 (PMR=10%) | 1 | **5** |
| 恶意客户端 (PMR=20%) | 2 | **10** |
| 良性客户端 | 9 | **40–45** |
| auto-γ (IID) | ≈10 | ≈ **10**（不变）或 **5** (PMR=20%) |
| auto-γ (α=0.1) | 8–47（单客户端极端波动） | ≈ **10**（多客户端分摊平滑） |
| 缩放后有效替换 | 严格 γ×w₀=1.0（全替换） | 每恶意客户端 γ×wᵢ < 1.0，但**集体** 实现替换 |
| 连续攻击下级联发散概率 | **高** | **极低**（45 个良性客户端梯度多样性形成缓冲） |

**关键推导**：N=50, PMR=20% 时，auto-γ = total_samples / (10 个恶意客户端总样本) = 50000/10000 = **5**。γ=5 远低于 P1 中触发 NaN 的阈值（γ=22.94）。40 个良性客户端在每轮聚合中提供充足的正则化。

#### 2.2.3 持续攻击对 VeriFL 防御的意义

- **持续攻击是防御算法的标准化压力测试**。在持续攻击下仍能有效防御，才能证明 VeriFL 的价值。
- N=50 + 持续攻击下，5–10 个恶意客户端的 γ≈5–10 缩放幅度较温和，恶意更新与良性更新的 norm 差距更小。**这才是检测算法的真正考验**——发现 "差异不大" 的恶意更新。
- 与之对比，N=10 + 单轮 + γ≈22 的配置下恶意更新极度异常，任何基于 norm/距离的防御都能轻松检测，无法展示 VeriFL 独有优势。

#### 2.2.4 配置变更

| 配置项 | P1.5 值 | 本方案 |
|:------|:-------|:------|
| `attack_training_rounds` | `[99]` (CIFAR-10) / `[49]` (MNIST) | **不设置（即每轮攻击）** |

当 `attack_training_rounds` 为 `null` 或未设置时，`ModelReplacementBackdoorAttack.attack_model()` 在每轮都执行缩放——这是代码的默认行为，无需修改攻击框架代码。

---

### 2.3 D-3：保持全参数缩放（不排除 BN）

这是与 v1 方案最大的分歧，经两轮独立分析后推翻了原有建议。

#### 2.3.1 v1 为什么建议排除 BN

P1.5 实验中 CIFAR-10 α=0.1 seed=0 出现 NaN（RC-1），直接原因是 auto-γ=22.94 对 ResNet18 的 20 层 BN `running_var` 极端放大，导致某些通道方差为负或极小，前向传播中 $1/\sqrt{var+\epsilon}$ 爆炸。v1 方案因此建议将 `should_scale_param()` 改为与 `is_weight_param()` 对齐，排除 `running_mean` 和 `running_var`。

#### 2.3.2 为什么在 N=50 下不再需要

NaN 发生的**必要条件组合**是：

1. **1 个恶意客户端**的极端局部 BN 统计量被完整注入全局模型
2. 该客户端仅有 2136 个极度偏斜样本（Dirichlet α=0.1, seed=0, N=10）
3. 某些通道 `running_var` 接近 0（该客户端几乎没见过这些类别），前向传播除以 $\sqrt{var}$ 导致爆炸
4. auto-γ = 22.94（异常偏高）

N=50 下上述条件全部自然消解：

| 条件 | N=10 (P1.5) | N=50 |
|:-----|:-----------|:-----|
| 恶意客户端 BN 来源 | **1 个**客户端的 2136 偏斜样本 | **5–10 个**客户端各 ~1000 样本，集体均值 |
| 类别覆盖 | 1 个 Dirichlet 分区，可能仅 1–2 类 | 5–10 个随机分区合并，类别覆盖显著更广 |
| 某通道 rv≈0 概率 | **高**（单客户端） | **极低**（多客户端均值平滑） |
| auto-γ | 8–47（极端波动） | ≈5–10（多客户端分摊） |

#### 2.3.3 排除 BN 反而会损害攻击效果

auto-γ 的设计目标是 $\gamma \times w_{\text{mal}} = 1.0$，即聚合后全局模型被恶意模型**完全替换**。如果排除 BN：

- **weight/bias** 被 γ 缩放 → 聚合后替换为恶意模型的 weight/bias ✓
- **BN stats** 未缩放 → 聚合后仍是全局模型的 BN stats（99% 来自良性客户端）✗

这造成 **内部不一致（internal inconsistency）**：恶意客户端的 weight/bias 是在自己的局部 BN stats 下训练的，但推理时全局模型用的是良性客户端主导的 BN stats。这种分布偏移会同时降低 ASR 和 clean acc。

#### 2.3.4 社区对标

| 论文 | 缩放范围描述 | 是否区分 BN |
|:-----|:-----------|:----------|
| FLTrust (NDSS'21) | "scales its local model update" | **否** |
| FLAME (USENIX Sec'22) | "model replacement attack" | **否** |
| FLDetector (S&P'22) | "scaling factor is set to 100" | **否** |
| FLAD (TDSC'25) | "weight factor γ=n" | **否** |
| FilterFL (CCS'25) | "scaled up by a factor of 10" | **否** |
| Fang 2025 (TDSC) | 引用原文 scaling | **否** |

**6/6 篇论文无一区分 learnable params 与 BN buffers**。全部描述为对 "entire model" 或 "model update" 进行缩放。保持全参数缩放是最干净的社区对标。

#### 2.3.5 保底措施

虽然评估 N=50 下 NaN 概率极低（<5%），但仍设置保底：

1. 第一批实验跑完后检查所有 JSONL 的 loss 列，grep NaN
2. 如果某个极端 seed 仍出现 NaN（概率 <5%），**针对性地对该组加 gamma_cap = N**，而非修改全局缩放逻辑。这保持了绝大多数实验的完整社区对标，仅对极端个案做最小侵入处理
3. 在论文 Implementation Details 中说明："In rare cases where extreme Dirichlet partitions cause numerical instability, we cap γ at N." 一句话足以覆盖

#### 2.3.6 代码变更

**对 `utils.py` 的 `should_scale_param()`：不做任何修改。** 保持原状。

---

### 2.4 D-4：Trigger Value 方案 B（代码层自适应转换）

#### 2.4.1 问题回顾

当前 `trigger_value = 1.0` 是在 **归一化后空间** 设置的。由于 MNIST 和 CIFAR-10 的 Normalize() 参数不同，same value = 1.0 对应不同的原始像素：

| 数据集 | Normalize 参数 | trigger=1.0 对应原始像素 | 社区标准（纯白=1.0）对应归一化后值 | 对比度比 |
|:------|:-------------|:------------------:|:----------------------------:|:-------:|
| MNIST | μ=0.1307, σ=0.3081 | **0.44**（暗灰） | **2.8215** | 35% |
| CIFAR-10 (R/G/B) | μ=(0.4914,0.4822,0.4465), σ=(0.2023,0.1994,0.2010) | (0.67,0.68,0.65)（中灰） | **(2.514, 2.596, 2.754)** | ~40% |

P1.5 MNIST 全部 20 组 ASR 不达标（median 最高 0.54），正是因为 trigger 对比度仅为社区标准的 35%。

#### 2.4.2 社区 trigger 定义方式

所有 6 篇参考论文的 trigger 均在 **原始像素空间** 定义：

| 论文 | Trigger 描述 | 空间 |
|:-----|:-----------|:----:|
| FLTrust | "same pattern trigger in [Bagdasaryan]" | 像素 |
| FLAD | "8×8 square pixel block" / "8×8 black block" | **明确像素** |
| FilterFL | "white stripe" / "color is red rather than white" | **明确像素** |
| FLDetector | "trigger patterns same as original papers" | 像素 |
| FLAME | 沿用原文 source code | 像素 |
| Fang 2025 | 引用原文 | 像素 |

#### 2.4.3 方案 B 设计

**核心思想**：config 中 `trigger_value` 的含义从 "归一化后空间的值" 改为 "原始像素空间的值 [0, 1]"。代码在注入 trigger 时根据数据集的 normalization 参数自动转换。

**config 不变**：`trigger_value: 1.0` 仍然是 1.0，但语义变为 "原始像素空间纯白色"——与社区完全一致。

**转换公式**：

$$\text{trigger\_normalized}[c] = \frac{\text{trigger\_value} - \text{mean}[c]}{\text{std}[c]}$$

| 数据集 | 通道数 | 归一化后 trigger 值 |
|:------|:-----:|:------------------|
| MNIST | 1 | $(1.0 - 0.1307) / 0.3081 = \mathbf{2.8215}$ |
| CIFAR-10 | 3 | $(2.5141,\ 2.5961,\ 2.7537)$ |

**需修改的代码位置**（2 处）：

1. **`verifl_trainer.py`**：训练时 trigger 注入。在 `__init__` 中根据 `self.args.dataset` 查表计算归一化后的 trigger tensor；在 `train()` 中使用该 tensor 注入（替换原来的 scalar）
2. **`asr.py`**：ASR 评估时 trigger 注入。`evaluate_asr()` 需增加 `dataset` 参数，内部做同样的转换

**归一化参数来源**：`data_loader.py` 中的硬编码常量，须保持一致：
- CIFAR-10: mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
- MNIST: mean=(0.1307,), std=(0.3081,)

#### 2.4.4 对 clean acc 的影响分析

Clean acc 评估路径 **不涉及 trigger**（仅在 ASR 评估路径注入）。trigger_value 改变仅影响恶意客户端训练时的后门样本强度。

更强的 trigger 反而 **有利于** clean acc：
- **强 trigger（2.82）**：在特征空间中形成显著的独立 shortcut，模型同时学好 clean 分类和 trigger→target 映射，互不干扰
- **弱 trigger（0.44）**：模型难以区分 trigger 和正常特征，gradients 互相干扰，损害正常学习

社区数据验证：FLTrust (NDSS'21) 在 MNIST 上使用像素空间 trigger + 每轮攻击 + 20% 恶意，报告 clean acc = **98%**（error=0.02），与无攻击基线几乎无差。

#### 2.4.5 对现有 CIFAR-10 结果的影响

CIFAR-10 的 P1.5 实验在 trigger=1.0（归一化后）下已达成 mean ASR ≥ 0.80（AC-P1.5-1 PASS，4/4 α 达标）。改为更强的 trigger 后 ASR 只会更高。无 regression 风险。

---

### 2.5 D-5：Auto-γ 保持现有实现，不加 gamma_cap

#### 2.5.1 原有 gamma_cap 建议的背景

`P1.5_AC失败根因分析.md` 中因 RC-1（α=0.1 seed=0, γ=22.94 导致 NaN）提出在 auto-γ 后加 γ 上限。但该建议基于 N=10 的特殊条件。

#### 2.5.2 N=50 下 gamma_cap 不再必要

auto-γ = total_samples / malicious_samples。在 N=50 下：

| PMR | 恶意客户端数 | 恶意总样本量（CIFAR-10 IID） | auto-γ | 需要 cap？ |
|:----|:----------|:------------------------:|:------:|:--------:|
| 10% | 5 | ≈ 5000 | ≈ **10** | 否 |
| 20% | 10 | ≈ 10000 | ≈ **5** | 否 |

即使 Dirichlet α=0.1 下个别恶意客户端数据极少，**5–10 个恶意客户端的总样本量**仍然稳定在一个合理范围，γ 不会出现极端值。

#### 2.5.3 不加 cap 的学术优势

gamma_cap 会导致 $\gamma \times w_{\text{mal}} < 1.0$，即模型替换退化为 **部分替换**。这削弱了攻击力度，使得实验无法回答 "在精确模型替换下防御是否有效" 这一核心问题。保持精确替换（γ×w=1.0）使攻击处于最强状态，能更好地展示防御的价值。

#### 2.5.4 NaN 风险的终局判定（v2.1 前置化）

P1/P1.5 中 NaN 的**根因**已完全定位：N=10 + Dirichlet α=0.1 → 单一恶意客户端数据极端偏斜 → auto-γ=22.94 → BN `running_var` 被极端放大 → 前向传播 $1/\sqrt{var}$ 爆炸。

N=50 下该根因的**每个必要条件**均已消解（详见 §2.3.2）。γ 仅由数据统计量决定（$\gamma = n_{\text{total}} / n_{\text{malicious}}$），与训练制度（$E=1$ vs $R_l=1$）无关。因此 **NaN 不是风险而是已关闭问题**。§七 中仍保留 B-NaN 作为极端情况保险，但预期概率为零。

#### 2.5.5 与社区 γ=n 公式的差异及对 AC 阈值的影响（v2.1 审计补充）

本项目的 auto-γ 公式与社区主流存在结构性差异：

| 来源 | 公式 | N=50, PMR=20% 下的 γ |
|:-----|:-----|:-----|
| **本项目** | $\gamma = n_{\text{total\_samples}} / n_{\text{malicious\_samples}}$ | ≈ **5** |
| Bagdasaryan 2020 原文 | $\gamma = n/m$（客户端数比） | **5** |
| Cao (FLTrust, NDSS'21) | $\gamma = n$（总客户端数） | **50** |
| Tang (FLAD, TDSC'25) | $\gamma = n$（总客户端数） | **50** |

**关键分析**：

1. **数学正确性**：本项目公式在 **sample-weighted FedAvg** 下是精准模型替换的正确推导——$\gamma = n_{\text{total}} / n_{\text{mal}}$ 确保恶意方聚合后有效权重 = 1.0。Bagdasaryan 原文的 $\gamma = n/m$ 等价于假设 uniform weighting 或 IID 等量分配。
2. **社区的 γ=n 是过度放大**：Cao/Tang 两篇论文的 γ=n=50（PMR=20%）= 本项目 γ≈5 的 **10 倍**。这导致社区报告的攻击效果不可直接比较——γ=50 下 CIFAR-10 clean acc 崩溃到随机猜测（Cao error=0.90），**攻击力远超精确模型替换**，已发生模型摧毁。
3. **AC 阈值影响**：社区在 γ=n 过度放大下报告的 ASR（FLTrust: 100%, FLAD: 99.8%）不能直接作为我们 γ≈5 下的对标基准。AC-A-1/A-2 阈值（≥0.80/≥0.70）已考虑了这一差距。但如果 γ≈5 下 ASR 显著低于预期，这是**公式差异导致的预期行为**，不是实现缺陷。

**社区 clean accuracy 的矛盾数据**（审稿人可能追问）：

| 数据集 | Cao (γ=n=10) | Tang (γ=n=50) | Fang (γ=n=100) |
|:------|:------------|:-------------|:--------------|
| MNIST | acc=98% ✅ | acc=94% ✅ | **acc=36% ❌** |
| CIFAR-10 | **acc=10% ❌** | acc=68% ✅ | **acc=10% ❌** |

同一攻击、同一 PMR=20%，不同论文的 clean accuracy 差异高达 **60 个百分点**。说明 γ 不是唯一影响因素——模型容量、训练轮数、LR 和局部训练强度都会影响攻击后 clean accuracy。这进一步支持我们在论文中**明确写出 γ 数值和公式来源**的必要性。

**论文写作要求**：Implementation Details 中须写明 "We follow Bagdasaryan et al.'s formulation for sample-weighted FedAvg: $\gamma = n_{\text{total}} / n_{\text{malicious}}$, yielding $\gamma \approx 5$ under $N=50, \text{PMR}=20\%$. This achieves exact model replacement under sample-count weighted aggregation."

#### 2.5.6 Non-IID 下 auto-γ 的 seed 间波动（v2.1 审计补充）

FedAvg 使用 sample-weighted 聚合（`weight_i = local_sample_number / total_samples`），Dirichlet α=0.1 分区下各客户端数据量可能差距较大。这导致：

1. **auto-γ 跨 seed 波动**：不同 seed 下恶意客户端获得的总样本量不同 → γ 值不同
2. **极端情况**：若恶意客户端恰好被分配极少数据 → γ 偏大；反之 γ 偏小

这不影响实验正确性，但需要在结果分析中报告 γ 跨 seed 的实际分布。D-13（§2.10）的 JSONL `gamma_actual` 字段正是为此服务。

---

### 2.6 D-6：PMR 仅 20%

| PMR | 恶意客户端数 (N=50) | 社区出处 | 用途 |
|:----|:-----------------|:--------|:-----|
| **20%** | 10 | FLTrust/FLAD/Fang 默认 | **主表数据**，直接对标社区 |

FLAD (TDSC'25) 的 Table III/IV 以 PMR=20% 为主实验；FLTrust 以 PMR=20% 为默认；Fang 2025 以 PMR=20% 为默认。4/6 篇参考论文以 20% 为主表配置。

**为什么不再测 PMR=10%**：
1. **社区主表对标**：仅 FilterFL (CCS'25) 以 10% 为默认，但 FilterFL 的实验配置（N=100, T=2000 CIFAR-10）与我们差异很大
2. **实验矩阵控制**：去掉 PMR=10% 可将实验总量从 54 降至 28，回收约 48% 的 GPU 时间
3. **论文聚焦性**：投稿论文仅需 PMR=20% 的 FedAvg baseline 即可完成 "无防御下攻击效果" 的论证。PMR=10% 为锦上添花，可在审稿人 rebuttal 阶段按需补充

---

### 2.7 D-7/D-8/D-9/D-10：沿用决策

| 编号 | 内容 | 说明 |
|:-----|:-----|:-----|
| D-7 | 攻击者本地超参 = 良性 | FLAD "as honest client"；社区共识 |
| D-8 | weight_decay CIFAR-10=1e-4, MNIST=0 | M1 冻结参数配方 |
| D-9 | 保留 MNIST | 6/6 篇论文测 MNIST 后门 |
| D-10 | JSONL 写入幂等化 | 数据管理 |

---

### 2.8 D-11：训练制度 E=1 (epoch-based)，不采用 $R_l=1$ (single-step)

#### 2.8.1 背景：社区存在两套并行标准

联邦学习 Byzantine-robust 文献中，本地训练制度存在两种不同的配置范式：

| 范式 | 代表论文 | 本地训练步数 | 通信轮次 $T$ | 理论基础 |
|:-----|:--------|:----------|:-----------|:--------|
| **理论优化** | FLTrust (NDSS'21), Fang 2025 (TDSC) | $R_l=1$（**单步** SGD） | 1500–2000 | FLTrust Theorem 1 显式要求 $R_l=1, \beta=1$ |
| **系统工程 (FedAvg)** | McMahan 2017, FLAD (TDSC'25) | $E \geq 1$ epoch | 20–100 | FedAvg 原文 $E \in \{1,5,20\}$ |

FLTrust 论文明确写道 (§V-A)：*"we set $R_l=1$, in which we can treat the product of the global learning rate $\alpha$ and the local learning rate $\beta$ as a single learning rate."* 这不是通用 FL 配置，而是 FLTrust 收敛证明 (Theorem 1) 的数学前提。

#### 2.8.2 为什么必须走 epoch-based ($E=1$)

**理由一：VeriFL 的 GA 机制依赖足够大的客户端更新。**

VeriFL Phase 1 Micro-GA 的适应度函数：$\text{fitness}(\boldsymbol{\alpha}) = \frac{1}{\text{val\_loss}(\boldsymbol{\alpha}) + \lambda \cdot \|\boldsymbol{w}_{\text{agg}}\| + \varepsilon}$

GA 通过搜索聚合权重 $\alpha$ 最小化验证损失来区分良性/恶意更新。这要求不同 $\alpha$ 组合产生的 $\text{val\_loss}$ 差异显著。

- $R_l=1$：每客户端更新 = 基于 64 个样本的**单步梯度**，更新范数极小且噪声主导 → $\text{val\_loss}$ 差异在浮点精度层面可能不可区分
- $E=1$：每客户端更新 = $\lceil 1000/64 \rceil = 16$ 步迭代的累积更新 → 更新幅度是 $R_l=1$ 的 ~16 倍，GA 有足够信噪比进行区分

**理由二：计算可行性。**

VeriFL GA **每轮无条件执行** 150 次前向传播（`pop_size=15 × generations=10`，代码已确认）。如果为匹配 FLTrust 的 $R_l=1$ 而将 $T$ 从 100 提升到 2000：

| 配置 | GA 前向传播总次数 | 估算 GA 总耗时 (4090) |
|:-----|:-------------:|:------------------:|
| $E=1, T=100$ | 15,000 | ~30 分钟 |
| $R_l=1, T=2000$ | 300,000 | ~10 小时 |

加上 MPI 通信开销（51 进程 × 2000 轮 × 双向参数传输 ~4.5GB/轮），单组实验运行时间将从 ~1 小时膨胀至数十小时。

**理由三：总计算量等价性。**

$$\text{Our gradient steps} = T \times \lceil n/b \rceil = 100 \times 16 = 1600 \text{ (CIFAR-10)}$$
$$\text{FLTrust gradient steps} = R_g \times R_l = 1500 \times 1 = 1500$$

总梯度步数相当，学术可比性有保障。

**理由四：FLAD (TDSC'25) 是我们最直接的对标论文（同为 N=50），其本地训练几乎必然采用 epoch-based。**

FLAD 仅 20 轮即达到 MNIST 95.2%、CIFAR-10 68.8%。FLTrust 用 $R_l=1$ 需要 T=2000 才达到类似精度。若 FLAD 也用 $R_l=1$，20 轮不可能收敛。

#### 2.8.3 配置确认

| 参数 | 值 | 说明 |
|:-----|:--|:-----|
| `epochs` (E) | **1** | 每轮每客户端训练 1 个完整 epoch |
| `comm_round` (T) | **100** (两个数据集统一) | CIFAR-10 100轮 = 社区可比；MNIST 从 50→100 以确保 N=50 下收敛 |

---

### 2.9 D-12：Dirichlet 分区移除 Boolean Cap，对齐社区标准（v2.1 审计新增）

#### 2.9.1 问题发现

代码审计发现 `data_loader.py` 中的 Dirichlet 分区（`_split_noniid`）包含一个**非标准的 Boolean cap 机制**：

```python
p = q * (len(idx_j) < len(dataset) / num_clients)
```

当客户端 $j$ 已有样本数 $\geq$ 每客户端平均值时，其对**剩余类别**的采样概率被置零。效果是抑制极端不平衡——α=0.1 下客户端间数据量差异远小于纯 Dirichlet 分布。

**该 cap 继承自 FedML 框架本身。** FedML 核心库的 `noniid_partition.py` 有一模一样的逻辑。ShieldFL 是照搬的，不是我们自己添加的。但 FedML 是工程框架，不是学术标准。

#### 2.9.2 社区对标

| 论文 | Dirichlet 实现 | 是否有 cap |
|:-----|:-------------|:----------|
| FLTrust (NDSS'21) | 标准 `np.random.dirichlet` | **否** |
| FLAD (TDSC'25) | "Dirichlet(α)" | **否**（论文层面） |
| Fang 2025 (TDSC) | "Dirichlet(α)" | **否**（论文层面） |
| FedSecurity (KDD'24) | FedML 框架 | **可能有**（同框架） |

社区论文描述中 **无一提及 cap**。6/6 篇论文写的都是标准 Dirichlet 分区。

#### 2.9.3 决策

**移除 Boolean cap，改用纯 Dirichlet 分区 + 最小样本保护。** 理由：

1. **一劳永逸的平台对齐**：移除 cap 后，所有未来实验（P2 攻击验证、M2 防御、M3 等）自动受益，不再需要在论文中解释 "我们的 Dirichlet 和社区不一样"。这是**基础设施层面的一次性修正**。
2. **消除可复现性风险**：论文中写 "Dirichlet α=0.1 non-IID"，但审稿人用纯 Dirichlet 复现时分布特征不同——这是学术信誉问题。保留 cap 需要在论文中声明并辩护；移除 cap 不需要任何额外说明。
3. **α=0.1 的异质性讨论失去精确基础**：我们的实验要对比 α=0.1 vs α=0.5 的异质性影响。如果底层 Dirichlet 本身被 cap 削弱，α=0.1 的实际分布向 α=0.5 方向偏移，两者的差异被人为缩小——这使得异质性维度的实验结论可信度下降。
4. **改动极小**：仅涉及 `_split_noniid` 函数中 ~6 行代码的删除/替换。

#### 2.9.4 移除 cap 后的安全保护

纯 Dirichlet α=0.1 下某些客户端可能获得极少样本（几十个甚至更少）。需要添加**最小样本数保护**：

| 保护机制 | 说明 |
|:--------|:-----|
| **最小样本阈值** | 如果任意客户端总样本数 < 10，则以不同的 Dirichlet 随机种子重新采样，直到满足条件（借鉴 FedML 框架版本的 `while min_size < 10` 外层循环机制） |
| **分区统计日志** | 分区完成后，输出每个客户端的样本数量和类别分布摘要到日志，用于事后审计和论文附录 |

#### 2.9.5 论文写作

移除 cap 后，论文中 **不需要** 任何额外声明——直接写 "We partition data among clients using Dirichlet distribution with concentration parameter α" 即可，与社区完全一致。

#### 2.9.6 验收标准

| AC | 条件 | 判定标准 |
|:---|:-----|:---------|
| **AC-C-4** | Dirichlet 分区正确性 | Phase 0 中打印 N=50, α=0.1, seed=0 下各客户端样本量统计：最大/最小值之比 > 3（确认 cap 已移除，异质性未被人为压缩）；所有客户端样本数 ≥ 10（最小样本保护生效） |

---

### 2.10 D-13：JSONL 结构化日志增补字段（v2.1 审计新增）

#### 2.10.1 问题发现

当前 JSONL 结构化日志（`metrics.py`）包含以下字段：

```json
{"round", "test_accuracy", "test_loss", "asr", "config_params", "git_commit", ...}
```

**缺失的关键字段**：

| 缺失字段 | 当前状态 | 影响 |
|:---------|:--------|:-----|
| `gamma_actual` | 仅在 `logging.info` 非结构化文本中输出 | AC-C-1 无法自动化验证 |
| `malicious_count` | 未记录 | 无法跨 seed 比对恶意客户端配置 |
| `trigger_value_normalized` | 未记录 | AC-C-3 无法自动化验证 |
| `max_asr` | 未计算 | 无法区分 FinalASR 是否低于历史峰值（见 D-14） |

#### 2.10.2 增补字段规格

| 字段名 | 类型 | 写入时机 | 示例值 |
|:------|:-----|:--------|:------|
| `gamma_actual` | float | 每轮攻击执行后 | `5.23` |
| `malicious_count` | int | 实验初始化 | `10` |
| `trigger_value_normalized` | list[float] | 实验初始化 | `[2.8215]` 或 `[2.514, 2.596, 2.754]` |
| `max_asr` | float | 每轮 ASR 评估后（running max） | `0.95` |

#### 2.10.3 代码变更

详见 §八 F-5。变更量约 15 行。

---

### 2.11 D-14：ASR 指标定义形式化（v2.1 审计新增）

#### 2.11.1 问题发现

本方案中多处使用 "ASR" 但未明确定义其统计含义。代码审计确认：当前实现**每轮**计算 ASR，但最终报告的是**末轮值**（FinalASR）。

#### 2.11.2 社区 ASR 报告惯例

| 论文 | ASR 定义 | 报告方式 |
|:-----|:--------|:--------|
| FLTrust (NDSS'21) | 200 轮后末轮 ASR | 表格单值 |
| FLAD (TDSC'25) | 末轮 ASR | 表格单值 + ASR vs round 曲线 |
| Fang 2025 (TDSC) | 末轮 testing error / ASR | 表格单值 |
| FedSecurity (KDD'24) | 末轮 ASR | 表格单值 |

**结论**：社区主流以 **FinalASR** 为主指标。

#### 2.11.3 为什么需要同时记录 MaxASR

在**持续攻击 + γ≈5**（相对温和的缩放）下，存在以下可能：

- Backdoor 在某中间轮次达到峰值 ASR，随后在良性客户端的持续更新下被部分覆盖
- FinalASR < MaxASR → 仅报告 FinalASR 可能低估攻击峰值强度

记录 MaxASR 作为**补充指标**可以：
1. 预防审稿人追问 "在训练过程中 backdoor 是否曾经更强"
2. 为后续防御阶段提供更完整的攻击画像
3. 帮助诊断 AC-A-7 因果性验证：如果 FinalASR 差距不足 0.30 但 MaxASR 差距达标，说明缩放因子确实有因果效应

#### 2.11.4 AC 判定基准

**AC-A-1/A-2 仍以 FinalASR 为判定基准**，与社区保持一致。MaxASR 仅记录，不纳入 PASS/FAIL 判定。

---

## 三、Gemini 建议总回溯

以下汇总历次交流中 Gemini 的全部主要建议及最终采纳情况，基于 6 篇 References 原文进行事实核查。

| 序号 | Gemini 建议 | 事实核查结果 | 采纳？ | 理由 |
|:-----|:----------|:----------|:-----:|:-----|
| G-1 | 用 GroupNorm 替换 BatchNorm | ❌ **无社区依据**：6/6 篇论文使用 BatchNorm | **不采纳** | 改变模型架构导致不可比 |
| G-2 | 将 MNIST 移出后门攻击测试集 | ❌ **事实错误**：6/6 篇论文在 MNIST 上测后门 | **不采纳** | MNIST 是社区标配 |
| G-3 | Single-shot 是社区标准 | ❌ **恰好相反**：5/6 篇论文每轮持续攻击 | **不采纳** | 改为每轮攻击 (D-2) |
| G-4 | 将 BN 统计量移出缩放区 | ⚠️ **方向有一定道理但不推荐**：N=50 下不需要 | **不采纳** | 保持全参数缩放 (D-3) |
| G-5 | 直接切 N=50 | ✅ **正确** | **采纳** | D-1 |
| G-6 | trigger 对比度需拉满 | ✅ **正确** | **采纳** | D-4 |
| G-7 | 方案 B（代码层自适应） | ✅ **正确** | **采纳** | D-4 |
| G-8 | $R_l=1$ 分析：不切换到 single-step；两套并行标准 | ✅ **核心结论正确**，"灾难3 NaN" 部分夸大（γ 由数据统计量决定，与 $R_l$ 无关）| **结论采纳** | D-11 |

---

## 四、冻结参数配方（N=50 版）

### 4.1 全局固定参数

以下参数继承自 M1 冻结配方，**不得修改**：

```yaml
training_type: cross_silo
federated_optimizer: FedAvg
client_optimizer: sgd
momentum: 0.9
server_momentum: 0.0
server_lr: 1.0
batch_size: 64
partition_method: hetero          # Dirichlet
target_label: 0
trigger_size: 3                   # 3×3 右下角 patch
trigger_value: 1.0                # 语义变更：原始像素空间值（代码自动转归一化）
backdoor_per_batch: 20
defense_type: none                # 无防御 FedAvg baseline
```

### 4.2 N=50 变更参数

| 参数 | P1.5 值 (N=10) | **本方案值 (N=50)** | 变更原因 |
|:-----|:-------------|:-----------------|:--------|
| `client_num_in_total` | 10 | **50** | D-1 |
| `client_num_per_round` | 10 | **50** | 全参与 |
| `scale_gamma` | auto | **auto** | 不变 |
| `attack_training_rounds` | [99] / [49] | **null**（每轮攻击） | D-2 |

### 4.3 PMR 与恶意客户端映射

| PMR | `ratio_of_poisoned_client` | `byzantine_client_num` (K) | 恶意 IDs |
|:----|:-------------------------|:--------------------------|:---------|
| 20% | 0.2 | **10** | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] |

> **v2.1 变更**：移除 PMR=10% 行。如需补充可按 K=5, IDs=[0,1,2,3,4] 配置。

### 4.4 任务线差异参数

| 参数 | CIFAR-10 + ResNet-18 | MNIST + LeNet-5 |
|:-----|:---------------------|:----------------|
| `learning_rate` | 0.01 | 0.01 |
| `weight_decay` | 0.0001 | 0.0 |
| `comm_round` (T) | 100 | **100** |
| `epochs` (E) | 1 | 1 |
| Dirichlet α 集合 | {0.1, 0.5, 100} | {0.1, 0.5, 100} |
| Seeds | {0, 1, 2} | {0, 1, 2, 3, 4} |

> **comm_round 说明**：v2 中 MNIST 为 T=50，v2.1 统一为 T=100。原因：N=50 下每客户端仅 1200 个 MNIST 样本（vs N=10 时 6000 个），需要更多通信轮次保证收敛。T=100 × E=1 × ⌈1200/64⌉ = 1900 总梯度步数，与 FLTrust 的 T=2000 × $R_l$=1 = 2000 步计算量相当。

> **α 集合说明**：P1.5 使用 {0.1, 0.3, 0.5, 100}。本方案精简为 {0.1, 0.5, 100}，覆盖 extreme non-IID / moderate non-IID / IID 三档。α=0.3 与 α=0.5 区分度有限，在 N=50 实验矩阵膨胀的背景下削去以控制实验总量。如审稿人要求可补充。

> **Seeds 说明**：CIFAR-10 用 3 seeds（训练成本较高），MNIST 用 5 seeds（训练成本低且固有方差大）。如果 CIFAR-10 实验方差过大，可扩展到 5 seeds。

---

## 五、实验矩阵

### 5.1 主实验矩阵

| 数据集 | 模型 | PMR | α 集合 | Seeds | 攻击轮 | 实验数量 |
|:------|:-----|:----|:------|:------|:------|:-------:|
| CIFAR-10 | ResNet-18 | 20% | {0.1, 0.5, 100} | {0,1,2} | 每轮 | 9 |
| MNIST | LeNet-5 | 20% | {0.1, 0.5, 100} | {0,1,2,3,4} | 每轮 | 15 |
| **小计** | | | | | | **24** |

> **v2.1 变更**：移除 PMR=10% 实验组（24 组）。4/6 篇参考论文以 PMR=20% 为主表，10% 为补充。若审稿人要求可在 rebuttal 阶段补充 PMR=10% 数据。

### 5.2 控制组

| 数据集 | PMR | α | Seed | γ | 目的 | 实验数 |
|:------|:----|:--|:-----|:--|:-----|:-----:|
| CIFAR-10 | 20% | 100 | 0 | 1 | γ=1 因果性控制 | 1 |
| MNIST | 20% | 100 | 0 | 1 | 同上 | 1 |
| CIFAR-10 | 0% | 0.5 | 0 | — | 无攻击 FedAvg 基线 | 1 |
| MNIST | 0% | 0.5 | 0 | — | 同上 | 1 |
| **小计** | | | | | | **4** |

> **v2.1 变更**：移除 PMR=10% 的 γ=1 控制组（2 组）。因果性验证只需在 PMR=20% 上完成即可。

### 5.3 实验总量

**24 + 4 = 28 组实验**

> 对比 v2 的 54 组，减少 48%。主要原因：移除 PMR=10% 全部实验组（24 主实验 + 2 控制组）。仅保留 PMR=20% 主力配置，与社区主表精确对标。

---

## 六、验收标准（AC）

### 6.1 设计原则

AC 设计遵循以下原则：
1. **社区可比性优先**：阈值参考 FLTrust/FLAD 在 FedAvg 无防御下的报告数据
2. **统计严谨性**：多 seed 取 median 或 mean，避免单 seed 异常值主导
3. **分层验收**：代码正确性 AC 与攻击效果 AC 分离

### 6.2 代码正确性 AC

| AC | 条件 | 判定标准 | 数据来源 |
|:---|:-----|:--------|:--------|
| **AC-C-1** | auto-γ 日志正确 | γ×w_mal = 1.0 ± 0.01 | JSONL `gamma_actual` 字段（D-13） |
| **AC-C-2** | γ=1 控制组不触发 auto 路径 | 日志中 0 条 auto-gamma | JSONL `gamma_actual` = 1.0 |
| **AC-C-3** | Trigger 值在归一化后空间正确 | MNIST trigger=2.82±0.01, CIFAR-10 每通道值与公式一致 | JSONL `trigger_value_normalized` 字段（D-13） |
| **AC-C-4** | Dirichlet 分区正确性（cap 已移除） | N=50, α=0.1 下客户端间最大/最小样本量之比 > 3 且所有客户端 ≥ 10 样本 | Phase 0 S0-5 分区统计日志（D-12） |

> AC-C-3 为新增，用于验证方案 B 的 trigger 自适应转换正确性。
>
> **v2.1 审计补充**：AC-C-1/C-3 的判定数据源从非结构化 `logging.info` 升级为 JSONL 结构化字段，确保 28 组实验可批量自动化验证。详见 D-13（§2.10）。

### 6.3 攻击效果 AC

#### 6.3.1 社区参考数据

| 论文 | 数据集 | 条件 | FedAvg ASR |
|:-----|:------|:-----|:----------|
| FLTrust (NDSS'21) | MNIST | PMR=20%, λ=n=10, 每轮 | **1.00** |
| FLAD (TDSC'25) | MNIST | PMR=20%, γ=n=50, 每轮 | **0.837** |
| FLTrust (NDSS'21) | CIFAR-10 | PMR=20%, λ=n=10, 每轮 | — (仅报告 error) |
| FLAD (TDSC'25) | CIFAR-10 | PMR=20%, γ=n=50, 每轮 | **0.998** (non-IID q=0.5) |

FedAvg 无防御 baseline 下，社区报告 ASR 普遍 ≥ 0.80。

> **⚠️ v2.1 审计关键提醒**：上述社区数据的 γ=n（10 或 50）显著高于本项目的 auto-γ≈5（sample-weighted 精确替换公式）。社区 γ=n 属于 **过度放大**（γ/γ_exact 比值为 2–10×），ASR 因此被高估。AC-A-1/A-2 阈值已设在低于社区报告值的保守水位，但仍需在实验后根据实际 γ 值做结果解读。详见 §2.5.5。

> **ASR 指标定义**：本方案中所有 AC 中的 "ASR" 均指 **FinalASR（末轮值）**，与社区报告惯例一致。MaxASR（历史峰值）作为补充指标记录但不纳入 PASS/FAIL 判定。详见 D-14（§2.11）。

#### 6.3.2 验收标准定义

| AC | 条件 | 判定标准 | 社区依据 |
|:---|:-----|:--------|:--------|
| **AC-A-1** | CIFAR-10 PMR=20% ASR | ≥2/3 α 的 mean ASR (3 seeds) ≥ 0.80 | FLAD CIFAR-10 FedAvg ASR=0.998（γ=50）；本项目 γ≈5 取保守阈值 |
| **AC-A-2** | MNIST PMR=20% ASR | ≥2/3 α 的 median ASR (5 seeds) ≥ 0.70 | FLAD MNIST FedAvg ASR=0.837（γ=50）；LeNet5 vs FLAD CNN 差异容限，取 0.70 |
| **AC-A-3** | 全部实验数值稳定性 | 0/28 组出现 loss=NaN | N=50 下 γ≈5，NaN 必要条件已消解（§2.5.4） |
| **AC-A-4** | CIFAR-10 无攻击基线 clean acc | α=0.5, seed=0, 100 轮后 test acc ≥ 70% | M1 基线 78–82%（N=10）；N=50 下每客户端数据量降至 1/5，保守下调 |
| **AC-A-5** | MNIST 无攻击基线 clean acc | α=0.5, seed=0, 100 轮后 test acc ≥ 95% | M1 基线 98–99%（N=10）；MNIST 收敛容易，N=50 影响有限 |
| **AC-A-6** | CIFAR-10 攻击后 clean acc | 所有非 NaN 组 clean acc > 10% | FLTrust γ=100：10%；FLAD γ=50：68%；Fang γ≈100：10%。本项目 γ≈5 预期**远高于** 10%（见下方说明） |
| **AC-A-7** | 因果性 ΔASR | auto-γ vs γ=1 的 ΔASR ≥ 0.30（≥ 1/2 控制组） | 证明缩放是 ASR 的因果因素；无直接社区先例，阈值基于 M2 实测 γ=1 ASR≈0.75–0.88 |

> **v2.1 变更**：移除 AC-A-2 (CIFAR-10 PMR=10%) 和 AC-A-4 (MNIST PMR=10%)，重新编号。因果性控制组从 4 组减为 2 组（仅 PMR=20%），判定标准相应从 "≥2/4" 调整为 "≥1/2"。
>
> **AC-A-6 审计补充说明**：社区在 γ=n 过度放大下，CIFAR-10 clean accuracy 表现高度不一致（Cao: 10%, Tang: 68%, Fang: 10%）。本项目 γ≈5 是**精确模型替换**（非过度放大），clean accuracy 预期**显著高于** 10% 阈值。如果实际 clean acc 偏低（如 <40%），需进一步排查是否存在其他问题而非简单 PASS。
>
> **AC-A-2 阈值说明**：
> - MNIST ASR 阈值低于 CIFAR-10，原因：(1) LeNet5 仅 61K 参数，后门嵌入空间有限；(2) MNIST 100 轮后几乎完全收敛，1 epoch 后门训练的梯度信号较弱。但改用像素空间触发器后，ASR 应显著提升（P1.5 中 trigger=1.0 归一化后 → median最高 0.54；改为 2.82 预期大幅提升）。
> - 如果 MNIST PMR=20% median ASR 仍不达标（<0.70），启动后备方案（见 §七）。

---

## 七、后备方案

如果主实验某些 AC 不通过，按以下顺序逐级升级：

### 7.1 MNIST ASR 不达标

| 后备等级 | 措施 | 代码支持 | 学术合理性 |
|:--------|:-----|:--------|:---------|
| B-1 | `attacker_epochs: 5` | 代码已支持（`verifl_trainer.py` L120-123） | 白盒攻击者可控制本地超参；Bagdasaryan 原文用 E=6 |
| B-2 | `attacker_lr: 0.05` | 代码已支持 | Bagdasaryan 原文恶意 lr=0.05 |
| B-3 | `backdoor_per_batch: 40` | 直接配置 | FLAD 80% 后门注入率远高于我们的 31% (20/64) |
| B-4 | 增大 `trigger_size` 到 5×5 | 直接配置 | FLAD 使用 8×8；FilterFL 使用条纹 pattern |

**选择原则**：每次仅改一个变量。优先 B-1（增加攻击者训练量，有原文支持），再 B-3（增加后门样本比例，有 FLAD 支持）。

### 7.2 CIFAR-10 NaN（预期概率为零，保险性保留）

> **前置判定**：N=50 下 auto-γ≈5–10，NaN 的所有必要条件已消解（详见 §2.5.4）。本节仅作极端情况保险。

| 后备等级 | 措施 | 说明 |
|:--------|:-----|:-----|
| B-NaN-1 | 对出问题的特定 seed 加 `gamma_cap = N` | 最小侵入，仅影响该组 |
| B-NaN-2 | 更换 seed | 排除极端 Dirichlet 分区 |

### 7.3 收敛不足（N=50 下每客户端仅 1000 样本）

| 后备等级 | 措施 | 说明 |
|:--------|:-----|:-----|
| B-C-1 | 增加 `epochs` 到 2 | 每轮更多本地训练（优先于增加轮次，因为通信成本更低） |
| B-C-2 | 增加 `comm_round` 到 150–200 | 给模型更多轮次收敛 |

---

## 八、需要修改的代码清单

本节仅列出 **修改点** 和 **学术意图**，不包含具体代码实现。工程同学根据本节的行为规格实施。

### 8.1 F-1：Trigger 自适应转换（对应 D-4）

**影响文件**：2 个

| 文件 | 修改类型 | 行为变更 |
|:-----|:--------|:--------|
| `verifl_trainer.py` | 修改 | `__init__` 中根据 `self.args.dataset` 查表获取 mean/std，将 `trigger_value`（像素空间）转为归一化空间的 per-channel tensor。`train()` 中用该 tensor 注入 trigger（替代原来的 scalar） |
| `asr.py` | 修改 | `evaluate_asr()` 增加 `dataset` 参数，函数内部进行相同的像素→归一化转换后注入 trigger |

**归一化参数查表**（须与 `data_loader.py` 保持一致）：

| 数据集 key | mean | std |
|:----------|:-----|:----|
| `"cifar10"` | (0.4914, 0.4822, 0.4465) | (0.2023, 0.1994, 0.2010) |
| `"mnist"` | (0.1307,) | (0.3081,) |

**关键行为约束**：
- `trigger_value: 1.0` 在 config 中 **含义不变**，但语义更新为 "原始像素空间中的纯白色"
- 对未知数据集（key 不在查表中），应 fallback 到直接使用 `trigger_value` 作为归一化后值（向后兼容）
- 日志必须输出转换后的 per-channel trigger 值，用于 AC-C-3 验证

### 8.2 F-2：`evaluate_asr()` 调用方传参（对应 D-4）

所有调用 `evaluate_asr()` 的位置需要传入 `dataset` 参数。工程同学需 grep 找到全部调用点并更新。

### 8.3 F-3：JSONL 写入幂等化（对应 D-10）

`run_experiment.sh` 中实验启动前，如果目标 JSONL 已存在，先备份到 `archive/` 子目录再新建。避免追加写入导致数据膨胀。

### 8.4 F-4：N=50 配置更新

`run_experiment.sh` 的默认参数或调用脚本需支持 N=50。具体包括：
- `--clients 50`
- `--attack_rounds` 不设置（每轮攻击）
- `--comm_round 100`（MNIST 和 CIFAR-10 统一为 100，v2.1 变更）
- GPU mapping 支持 51 个进程

### 8.5 F-5：JSONL 结构化日志增补字段（对应 D-13，v2.1 审计新增）

**影响文件**：2 个

| 文件 | 修改类型 | 行为变更 |
|:-----|:--------|:--------|
| `metrics.py` | 修改 | JSONL 输出增加 `gamma_actual`、`malicious_count`、`trigger_value_normalized`、`max_asr` 四个字段 |
| `shieldfl_aggregator.py` 或攻击模块 | 修改 | 将 `gamma_actual` 值从攻击模块传递到 metrics 记录层（当前仅在 `logging.info` 中输出） |

**字段规格**：

| 字段名 | 类型 | 写入时机 | 示例值 | 用途 |
|:------|:-----|:--------|:------|:-----|
| `gamma_actual` | float | 每轮攻击执行后 | `5.23` | AC-C-1 自动化验证 |
| `malicious_count` | int | 实验初始化（每条记录重复写入） | `10` | 跨 seed 配置一致性校验 |
| `trigger_value_normalized` | list[float] | 实验初始化（每条记录重复写入） | `[2.8215]` | AC-C-3 自动化验证 |
| `max_asr` | float | 每轮 ASR 评估后（running max） | `0.95` | D-14 补充指标 |

**关键行为约束**：
- `max_asr` 须维护一个跨轮的 running maximum 变量，每轮更新为 `max(max_asr_so_far, current_asr)`
- 无攻击基线实验中，`gamma_actual` 字段写入 `null`
- γ=1 控制组中，`gamma_actual` 字段写入 `1.0`

### 8.6 F-6：Dirichlet 分区移除 Cap + 最小样本保护（对应 D-12，v2.1 审计新增）

**影响文件**：1 个

| 文件 | 修改类型 | 行为变更 |
|:-----|:--------|:--------|
| `data_loader.py` | 修改 | (1) 移除 `_split_noniid` 中的 Boolean cap 逻辑；(2) 添加最小样本数保护（重采样机制）；(3) 分区完成后输出客户端样本分布统计日志 |

**修改行为规格**：

1. **移除 cap**：删除当前函数中将已达平均样本量的客户端概率置零的 Boolean 乘法逻辑，保留纯 Dirichlet 采样→归一化→按比例分配的流程
2. **最小样本保护**：添加外层重试机制——如果分区后存在样本数 < 10 的客户端，以递增的随机种子偏移重新执行 Dirichlet 采样，最多重试 100 次（参考 FedML 框架 `noniid_partition.py` 的 `while min_size < 10` 模式）
3. **分区统计日志**：分区完成后，输出 summary 日志，包含每个客户端的样本数量和类别分布摘要

日志输出格式：

```
[DataPartition] N=50, alpha=0.1, seed=0 (retries=0)
  Client 0: 2341 samples, classes=[0:312, 1:45, 3:1984]
  Client 1: 187 samples, classes=[2:143, 7:44]
  ...
  Stats: min=87, max=3201, mean=1000, std=742
```

**关键行为约束**：
- 函数签名和返回值不变（仍返回 `client_idcs`），下游代码无需修改
- 重试仅改变 Dirichlet 采样的随机种子偏移，不改变 seed 本身的含义
- 必须在日志中记录实际重试次数，用于 AC-C-4 验证

### 8.7 不需要修改的代码

| 文件/函数 | 说明 |
|:---------|:-----|
| `utils.py` 的 `should_scale_param()` | 保持原状（D-3） |
| `model_replacement_backdoor_attack.py` 的 auto-γ | 保持原状（D-5） |
| 模型定义文件 (resnet18.py, lenet5.py) | 不改 |
| FedAvg 聚合逻辑 | 不改 |

---

## 九、实验执行顺序

### 9.1 Phase 0：前置验证（smoke test）

| 步骤 | 内容 | 目的 | 实验数 |
|:-----|:-----|:-----|:-----:|
| S0-1 | N=50 无攻击 FedAvg，CIFAR-10 α=0.5 seed=0，跑 10 轮 | 确认 51 进程启动无 OOM，显存足够 | 1 |
| S0-2 | N=50 无攻击 FedAvg，MNIST α=0.5 seed=0，跑 5 轮 | 同上 | 1 |
| S0-3 | 检查 trigger 自适应转换日志 | AC-C-3 预验证 | 0 (复用 S0-1/S0-2 日志) |
| S0-4 | N=50 **有攻击** FedAvg，CIFAR-10 α=100 seed=0，跑 **3 轮** | 端到端攻击管线验证：gamma_actual 写入 JSONL、trigger 归一化值正确、ASR 计算管线不崩溃 | 1 |
| S0-5 | 打印 Dirichlet 分区统计（α=0.1, seed=0, N=50），验证 cap 已移除且最小样本保护生效 | AC-C-4 验证（不占 GPU） | 0 |

**通过标准**：
1. 无 OOM、无 crash
2. Trigger 日志值正确（AC-C-3 预验证）
3. **（v2.1 新增）** S0-4 的 JSONL 中 `gamma_actual` 字段存在且值 ≈ 5.0 ± 1.0
4. **（v2.1 新增）** S0-4 的 JSONL 中 `trigger_value_normalized` 字段存在且 CIFAR-10 值 ≈ [2.51, 2.60, 2.75]
5. **（v2.1 新增）** S0-4 的 JSONL 中 `max_asr` 字段存在
6. **（v2.1 新增）** S0-5 的分区统计日志中：客户端间最大/最小样本量之比 > 3 且所有客户端样本数 ≥ 10（AC-C-4）

### 9.2 Phase 1：无攻击基线

| 步骤 | 内容 | 实验数 |
|:-----|:-----|:-----:|
| S1-1 | CIFAR-10 无攻击，α=0.5, seed=0, 100 轮 | 1 |
| S1-2 | MNIST 无攻击，α=0.5, seed=0, **100 轮** | 1 |

**通过标准**：AC-A-4（CIFAR-10 clean acc ≥ 70%）、AC-A-5（MNIST clean acc ≥ 95%）。

> **为什么需要重跑基线**：M1 基线在 N=10 下测定。N=50 下每客户端数据量变为 1/5，收敛行为可能不同。必须重新确认 FedAvg 在 N=50 下的基线性能。
> 
> 如果基线不达标，优先调整 `epochs` 到 2（后备 B-C-1），再增加 `comm_round`（后备 B-C-2）。

### 9.3 Phase 2：主实验

| 步骤 | 内容 | 实验数 |
|:-----|:-----|:-----:|
| S2-1 | CIFAR-10 PMR=20%, 3α × 3 seeds | 9 |
| S2-2 | MNIST PMR=20%, 3α × 5 seeds | 15 |

### 9.4 Phase 3：控制组

| 步骤 | 内容 | 实验数 |
|:-----|:-----|:-----:|
| S3-1 | γ=1 控制组（2 组：CIFAR-10/MNIST 各 1 组） | 2 |

### 9.5 Phase 4：AC 验收

从全部 28 组实验结果中提取数据，逐条判定 §六 中的 AC。

---

## 十、与 P1.5 AC 的对应关系

P1.5 的 AC 体系（AC-C-1~3, AC-P1.5-1~8）是为 N=10 + 单轮攻击设计的。切换到 N=50 + 持续攻击后，AC 体系需要升级。对应关系如下：

| P1.5 AC | 本方案 AC | 变化说明 |
|:--------|:---------|:--------|
| AC-C-1 (auto-γ 正确) | **AC-C-1** | 不变 |
| AC-C-2 (γ=1 不触发 auto) | **AC-C-2** | 不变 |
| — (新增) | **AC-C-3** | trigger 自适应转换验证 |
| AC-P1.5-1 (CIFAR-10 ASR) | **AC-A-1** | 仅 PMR=20% |
| AC-P1.5-2 (CIFAR-10 no NaN) | **AC-A-3** | 范围扩大到全部实验 |
| AC-P1.5-3 (clean acc > 30%) | **AC-A-6** | clean acc > 10% (排除 NaN) |
| AC-P1.5-4 (ΔASR ≥ 0.30) | **AC-A-7** | 控制组减为 2 组，判定 ≥1/2 |
| AC-P1.5-5 (MNIST median ASR) | **AC-A-2** | 仅 PMR=20%，阈值基于社区数据调整 |
| AC-P1.5-6 (MNIST IID ASR) | 合并入 AC-A-2 | 不再单设 IID 专项 AC |
| AC-P1.5-7 (MNIST no NaN) | 合并入 AC-A-3 | — |
| AC-P1.5-8 (因果性) | **AC-A-7** | 控制组 2 组 |

---

## 十一、风险登记表

| 风险 | 概率 | 影响 | 缓解方案 |
|:-----|:----:|:----:|:--------|
| N=50 OOM | 低 | 高 | Phase 0 smoke test 验证显存；必要时降低 batch_size 到 32 |
| N=50 收敛不足 | 中 | 中 | Phase 1 基线验证；后备 B-C-1/B-C-2 |
| CIFAR-10 α=0.1 NaN | **极低** (N=50, γ≈5) | 低 | 已关闭问题（§2.5.4）；保险性保留 B-NaN-1 |
| MNIST ASR 仍不达标 | 中 | 中 | 后备 B-1~B-4 逐级升级 |
| GPU mapping 配置不支持 51 进程 | 低 | 高 | 优先修改 gpu_mapping.yaml；或改用环境变量控制 |
| 持续攻击下 clean acc 过低 (<10%) | 低 | 低 | 本项目 γ≈5 为精确模型替换（非过度放大），攻击设计意图保持 clean acc 不受损。γ≈5 下崩溃到 <10% **不是正常现象**，须排查 BN 统计量、trigger 过拟合等工程问题。仅在 γ>>γ_exact 的过度放大下（如社区 γ=50–100）clean acc 崩溃才是预期行为 |
| γ≈5 下 ASR 低于 AC 阈值 | 中 | 中 | **（v2.1 新增）** 本项目 auto-γ≈5 远低于社区 γ=n=50 的过度放大。ASR 可能低于社区报告值，但仍应满足 AC-A-1/A-2 的保守阈值（0.80/0.70）。如不满足，先确认 trigger 转换（D-4）和 gamma（D-13 JSONL）是否正确，排除实现问题后启动后备方案 |
| Dirichlet 移除 cap 后极端偏斜客户端 | 低 | 低 | **（v2.1 新增）** 最小样本保护确保所有客户端 ≥ 10 样本（D-12）；极端偏斜是 Dirichlet α=0.1 的预期行为，与社区标准一致 |

---

## 十二、预期论文数据呈现

本方案的 28 组实验数据预期产出以下论文表格/图表：

### 12.1 Table X: Attack Effectiveness (No Defense)

| Dataset | Model | PMR | α=0.1 ASR | α=0.5 ASR | IID ASR | α=0.1 Acc | α=0.5 Acc | IID Acc |
|:--------|:------|:----|:----------|:----------|:--------|:----------|:----------|:--------|
| CIFAR-10 | ResNet-18 | 20% | — | — | — | — | — | — |
| MNIST | LeNet-5 | 20% | — | — | — | — | — | — |

此表直接对标 FLAD Table IV(a)，展示无防御下 Scaling Attack 的 baseline 效果。

### 12.2 后续 M2 防御实验的接入点

本方案建立的无防御 FedAvg baseline，将作为后续 VeriFL 及其他防御方法的 **攻击基线参照**。防御实验的实验矩阵将复用本方案的冻结参数、数据集、模型、PMR、α 配置，仅改变 `defense_type`。

---

## 附录 A：P1 + P1.5 教训总结

| 阶段 | 实验数 | 关键教训 |
|:-----|:-----:|:--------|
| P1 | 51 | γ=N 在 N=10 下连续 5 轮攻击导致级联 NaN；MNIST seed=1 ASR=0 为统计样本不足 |
| P1.5 | 35 | auto-γ 在极端 non-IID 下可产生 γ>20 → BN 级联 NaN（仅 1/12 组触发）；trigger=1.0 在归一化后空间仅等效 0.44 像素 → MNIST ASR 全部不达标 |
| **总结** | 86 | 攻防代码管线正确（AC-C-* 全部 PASS）；所有失败均为参数配置问题；N=10 是不可发表的实验规模 |

## 附录 B：6+1 篇参考论文核心参数快查

| 论文 | 会议 | N | PMR | γ 公式 | γ 实际值 | 攻击频率 | 本地训练 | T | MNIST ASR (FedAvg) | CIFAR-10 ASR (FedAvg) |
|:-----|:-----|:--|:----|:------|:--------|:--------|:--------|:--|:-------------------|:---------------------|
| FLTrust | NDSS'21 | 100 | 20% | $\lambda=n$ | **100** | 每轮 | **$R_l=1$** (single-step) | 2000/1500 | **1.00** | ~高 (error 报告) |
| FLAME | USENIX Sec'22 | 100 | 20%+ | 原文 | — | 每轮 | 未说明 | — | — | — |
| FLDetector | S&P'22 | 100 | 28% | $\gamma=100$ | **100** | 每轮 | 未说明 | — | 0.5% (有防御) | — |
| FLAD | TDSC'25 | 50 | 20% | $\gamma=n$ | **50** | 每轮 | **epoch-based** (E 未显式给出) | **20** | **0.837** | **0.998** |
| FilterFL | CCS'25 | 100 | 10% | $\gamma=10$ | **10** | 末段 10% | 未说明 | 100/2000 | — | — |
| Fang 2025 | TDSC | 100 | 20% | 引用原文 | — | 每轮 | 引用 FLTrust ($R_l=1$) | 2000/1000 | — | — |
| FedSecurity | KDD'24 | 10 | 10% | — | — | 每轮 | 未说明 | — | — | — |
| **本项目** | — | **50** | **20%** | $\gamma=n_\text{total}/n_\text{mal}$ | **≈5** | **每轮** | **E=1 epoch-based** | **100** | **待测** | **待测** |

> **v2.1 审计关键发现**：社区普遍使用 $\gamma=n$（总客户端数），远高于精确模型替换所需的 $\gamma=n/m$ 或本项目的 sample-weighted 公式 $\gamma \approx 5$。社区报告的 ASR 值需在此差异下理解——$\gamma=50$ 是精确替换 $\gamma=5$ 的 10 倍过度放大。  
> **v2.1 新增列**："γ 公式"、"γ 实际值"。FLTrust 系论文用 $R_l=1$+高 $T$；FLAD 用 epoch-based+极低 $T$(20)。我们的 $E=1, T=100$ 处于两者之间，总梯度步数 (1600) 与 FLTrust (1500) 相当。  
> **v2.1 新增行**：FedSecurity (KDD'24) — FedML 框架自身的基准论文，作为框架实现对标参考。

## 附录 C：代码现状基线

以下为当前需修改文件的核心函数签名和行为，供工程同学参考：

| 文件 | 函数/位置 | 当前行为 | 目标行为 |
|:-----|:---------|:--------|:--------|
| `verifl_trainer.py` L147-153 | `images[idx, :, -sz:, -sz:] = self._trigger_value` | 使用 scalar 值 (1.0) 统一所有通道 | 使用 per-channel normalized tensor |
| `asr.py` L53 | `images[:, :, -ts:, -ts:] = trigger_value` | 使用 scalar 值 | 使用 per-channel normalized tensor，需接收 dataset 参数 |
| `utils.py` L27-28 | `should_scale_param(k)` 排除 num_batches_tracked | — | **不改** |
| `model_replacement_backdoor_attack.py` L84-96 | auto-γ = total/malicious，无 cap | — | **不改** |
