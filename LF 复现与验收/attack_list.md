# VeriFL 实验攻击方法复现清单

> **用途**: 交付给工程同学的攻击方法复现清单
> **选择方法论**: 双轴驱动 — ①基线公平对比需求 × ②VeriFL 三阶段防御机理压测
> **生成日期**: 2026-04-03 v3
> **约束**: CCS 12 页篇幅，实验章节 ≈ 3 页，**6 个复现攻击 + 1 个自适应攻击 = 7 个实验条目**

---

## 0. 选择逻辑说明

### 双轴选择标准

每一个入选攻击必须**同时**满足两条轴：

- **轴 1 — 基线覆盖**: 该攻击在我们至少 2 篇核心基线论文的实验中出现，使得对比表格有意义
- **轴 2 — 阶段压测**: 该攻击对 VeriFL 三阶段防御中的**某个特定阶段**构成有针对性的威胁

不满足双轴的攻击不入选，不论它多"经典"。

### 为什么是 6+1

同领域 CCS / S&P / USENIX Sec 防御论文的攻击数量参考：
- FLTrust (NDSS'21): 5 攻击
- FLAME (USENIX Sec'22): 4 攻击
- FilterFL (CCS'25): 4 攻击
- DnC (S&P'22): 4 攻击
- FLAD (TDSC'25): 7 攻击

CCS 12 页 = 实验章节 ≈ 3 页 = 1 untargeted 表 + 1 targeted 表 + 消融 + 自适应。
6 个复现攻击（3 untargeted + 3 targeted）+ 1 个手工自适应攻击 = 可以在 3 页内完整呈现。

### 我们的防御基线（论文 Table 中的列）

| 基线 | 出处 | 选择理由 |
|------|------|---------|
| FedAvg | McMahan et al., AISTATS 2017 | 无防御 baseline |
| Multi-Krum | Blanchard et al., NeurIPS 2017 | 经典距离选择 |
| Trimmed Mean | Yin et al., ICML 2018 | 经典坐标统计 |
| Median | Yin et al., ICML 2018 | 与 Trimmed Mean 配对 |
| FLTrust | Cao et al., NDSS 2021 | **最直接竞争者** — 同用服务器数据 |
| FLAME | Nguyen et al., USENIX Sec 2022 | 聚类+DP 范式代表 |
| FLAD | Tang et al., IEEE TDSC 2025 | 神经网络特征+聚类；也用服务器数据 |
| FilterFL | Ren et al., CCS 2025 | **同场竞技** — CCS'25 最新防御 |

---

## 1. 最终攻击清单（6 个复现 + 1 个自适应）

### 精简矩阵：入选攻击 × 基线覆盖 × VeriFL 阶段映射

| # | 攻击 | 类型 | 基线覆盖 | VeriFL 压测目标 |
|---|------|-----|---------|---------------|
| 1 | Label Flipping | Untargeted | FLTrust / FLAD / Fang'25 (3/5) | Phase 1（梯度方向偏离 → 验证损失上升 → GA 降权） |
| 2 | Trim Attack | Untargeted | FLTrust / Fang'25 (2/5) | Phase 1 + 基线对比（AGR-tailored 代表，TrimMean 之克星） |
| 3 | Min-Max | Untargeted | Fang'25 (1/5)† | Phase 1 深度压测（AGR-agnostic 最强，在距离约束内最大化偏移） |
| 4 | Scaling Attack | Targeted | **全部 5/5** | **Phase 2 核心对手**（放大模长 → 锚点投影主防线） |
| 5 | DBA | Targeted | FLAME / FLAD / Fang'25 / FilterFL (4/5) | Phase 3 压测（分布式触发 → 需多轮聚合 → 动量惯性阻断） |
| 6 | Neurotoxin | Targeted | Fang'25 / FilterFL (2/5) | **Phase 3 终极测试**（注入 top-k 持久维度 → 挑战 §16.8 动量累积） |
| 7 | Adaptive Attack | Both | — (自研) | **全链路**（针对 §16.1 锚点信任 + §16.2 方向盲区 + §16.8 惯性陷阱） |

> † Min-Max 虽在基线覆盖表中仅 Fang'25 明确测试，但它是 **AGR-agnostic** 最强攻击（不假设聚合规则），是测试 VeriFL Phase 1 "验证集驱动搜索能否抵御优化型攻击"的唯一选择，不可替代。

### 逐一详述

#### ① Label Flipping（Untargeted）

| 属性 | 详情 |
|------|------|
| **原文** | Fang et al., "Local Model Poisoning Attacks to Byzantine-Robust Federated Learning", USENIX Security 2020 |
| **机制** | 恶意客户端将标签 $y$ 翻转为 $C-1-y$（CIFAR-10 中 $y \to 9-y$），用错误标签训练后上传 |
| **对 VeriFL 的意义** | Phase 1 基本测试：翻转标签使梯度方向偏离正确方向 → 聚合后验证损失升高 → GA 应自动降权。如果连这个都防不住，整个方案不成立 |
| **基线覆盖** | FLTrust (NDSS'21), FLAD (TDSC'25), Fang'25 均测试 |
| **复现要点** | 标签映射 $y \to C-1-y$；其余训练流程与良性客户端一致 |

#### ② Trim Attack（Untargeted, AGR-specific）

| 属性 | 详情 |
|------|------|
| **原文** | Fang et al., "Local Model Poisoning Attacks to Byzantine-Robust Federated Learning", USENIX Security 2020 |
| **机制** | 攻击者知道聚合规则为 Trimmed Mean，逆向优化恶意梯度使得 trim 后残留最大偏移 |
| **对 VeriFL 的意义** | VeriFL 不使用坐标级统计，理论上对 AGR-tailored 攻击天然免疫。此攻击验证这一优势——若 VeriFL 在 Trim Attack 下表现远优于 Trimmed Mean，证明"整体权重搜索"比"坐标统计"更抗优化攻击 |
| **基线覆盖** | FLTrust (NDSS'21), Fang'25 均测试 |
| **复现要点** | 需已知目标 AGR 为 Trimmed Mean；实现时参照 Fang et al. 开源代码 |

#### ③ Min-Max（Untargeted, AGR-agnostic）

| 属性 | 详情 |
|------|------|
| **原文** | Shejwalkar & Houmansadr, "Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for Federated Learning", NDSS 2021 |
| **机制** | 在不依赖任何聚合规则先验的条件下，优化恶意梯度使其与良性梯度的最大距离最小化（不被踢出），同时最大化对全局模型的伤害 |
| **对 VeriFL 的意义** | **Phase 1 深度压测**。Min-Max 刻意保持更新"看起来正常"（距离在合理范围内），这直接挑战 VeriFL 的验证集损失评估——如果恶意更新在距离上伪装良性，GA 的适应度函数能否通过验证损失发现异常？这是测试 Phase 1 "不依赖距离指标、纯靠效果评估"核心优势的关键攻击 |
| **基线覆盖** | Fang'25 测试；DnC (S&P'22) 的 DYN-OPT 为近似变体 |
| **复现要点** | 需要解内层优化问题；Shejwalkar & Houmansadr 开源了参考实现 |

#### ④ Scaling Attack（Targeted）

| 属性 | 详情 |
|------|------|
| **原文** | Bagdasaryan et al., "How To Back Door Federated Learning", AISTATS 2020 |
| **机制** | 单个恶意客户端在本地植入后门任务，训练后将整个模型更新乘以系数 $\gamma \gg 1$（如 $N/1$），使 FedAvg 聚合后全局模型被该客户端主导 |
| **对 VeriFL 的意义** | **Phase 2 核心对手**。这是锚点归一化投影（Anchor Projection）的直接测试：放大后的 $\|W_i\|_T$ 远大于锚点 $\|W_a\|_T$，缩放因子 $s_i = r_a / (\|W_i\|_T + \varepsilon) \ll 1$，恶意影响被压缩。同时 Phase 1 的 $\lambda=0.1$ 范数惩罚在适应度中形成第一道屏障 |
| **基线覆盖** | **全部 5/5 基线**均测试某种变体 — 最高覆盖率 |
| **复现要点** | 后门任务：选定触发器（像素 patch）+ 目标标签；放大系数参照原文设 $\gamma = N/K$ |

#### ⑤ DBA — Distributed Backdoor Attack（Targeted）

| 属性 | 详情 |
|------|------|
| **原文** | Xie et al., "DBA: Distributed Backdoor Attacks against Federated Learning", ICLR 2020 |
| **机制** | 将后门触发器拆分为 $K$ 个子触发器，分别分配给 $K$ 个恶意客户端。每个客户端只注入部分触发器，聚合后完整后门在全局模型中"拼合"生效 |
| **对 VeriFL 的意义** | **Phase 3 压测**。每个恶意客户端的单独更新偏离度较小（子触发器不等于完整后门），Phase 1 可能给出非零权重。后门效果需要多轮持续注入才能在全局模型中累积——这直接测试 Phase 3 动量平滑（$\beta = 0.9$）能否通过时序惯性阻止后门拼合。同时测试 §16.8 动量累积风险：如果每轮微量后门方向一致，$\mathbf{V}_t$ 是否会在该方向上持续累积 |
| **基线覆盖** | FLAME (USENIX Sec'22), FLAD (TDSC'25), Fang'25, FilterFL (CCS'25) 均测试 (4/5) |
| **复现要点** | 配置子触发器数量 = 恶意客户端数；每个客户端只注入自己的子触发器子集；推理时注入完整触发器计算 ASR |

#### ⑥ Neurotoxin（Targeted）

| 属性 | 详情 |
|------|------|
| **原文** | Zhang et al., "Neurotoxin: Durable Backdoors in Federated Learning", ICML 2022 |
| **机制** | 在后门训练后，投影恶意更新到 top-k 坐标（按全局模型梯度幅值排序），使后门信号集中在更新频率最低的维度 → 后续良性聚合不容易覆盖这些维度 → 后门持久性大幅提高 |
| **对 VeriFL 的意义** | **Phase 3 终极测试**。Neurotoxin 专门设计为"在聚合后存活更久"，直接挑战动量平滑的衰减速率。$\beta = 0.9$ 意味着历史方向需 ≈10 轮才衰减到 35%，而 Neurotoxin 注入的低频维度本就不容易被良性更新覆盖。同时，由于模长保持正常范围，Phase 2 锚点投影对其**无效**（§16.2 方向盲区），防御完全依赖 Phase 1 + Phase 3 |
| **基线覆盖** | Fang'25, FilterFL (CCS'25) 均测试 |
| **复现要点** | 需读取全局模型参数计算 top-k 掩码；k 值参照原文设定；后门训练后乘以掩码再上传 |

#### ⑦ VeriFL-specific Adaptive Attack（自适应，非复现）

| 属性 | 详情 |
|------|------|
| **来源** | 自研，不是复现已有工作 |
| **设计依据** | VeriFL 的三个已知局限（algorithm.md §16） |
| **必要性** | CCS 审稿人**必然追问**: "如果攻击者知道你的防御细节，能否绕过？" 没有自适应攻击 = 几乎确定 reject |

**三阶段攻击面分析**：

| VeriFL 阶段 | 已知弱点 (§16) | 自适应攻击方向 |
|------------|----------------|--------------|
| Phase 1 GA 搜索 | §16.1 锚点可信性：若恶意更新在验证集上"意外低损失"则获高 $\alpha$ | 构造使验证集损失低但含隐蔽后门的更新（代理目标优化：$\min L_{val} + \lambda_{atk} \cdot L_{backdoor}$）|
| Phase 2 锚点投影 | §16.2 方向盲区：只约束模长不约束方向 | 保持 $\|W_i\|_T \approx \|W_a\|_T$ 使 $s_i \approx 1$，在方向上注入偏差 |
| Phase 3 动量平滑 | §16.8 动量累积：持续同向偏移可被 $\mathbf{V}_t$ 累积 | Neurotoxin 变体：注入低频持久维度，利用 $\beta=0.9$ 的长记忆窗口 |
| **全链路** | 三个弱点可联合利用 | 同时满足：低验证损失 + 正常模长 + 持久维度注入 |

**推荐实验设计**: 至少实现两个变体：
1. **Phase-1-aware**: 优化恶意更新使 $F(W_{malicious})$ 接近 $F(W_{benign})$，绕过 GA 检测
2. **Full-pipeline**: 联合优化验证损失 + 模长约束 + top-k 持久维度，三阶段同时攻击

---

## 2. VeriFL 三阶段防御 × 攻击映射总览

```
                    攻击
                    ─────────────────────────────────
                    LF   Trim  MinMax  Scale  DBA  Neuro  Adaptive
Phase 1 (GA搜索)    ●●   ●●    ●●●    ●      ●    ●      ●●●
Phase 2 (锚点投影)  ─    ─     ─      ●●●    ─    ─      ●●
Phase 3 (动量平滑)  ●    ─     ─      ●      ●●●  ●●●    ●●●
```

- ●●● = 该攻击是该阶段的**核心压测**
- ●● = 该攻击对该阶段有中等测试作用
- ● = 该攻击在该阶段有边际测试作用
- ─ = 该攻击不测试该阶段

**每个阶段都有至少一个 ●●● 的核心对手，三阶段防御的完整性得到验证。**

---

## 3. 被裁掉的攻击及理由

| 攻击 | 裁掉理由 |
|------|---------|
| Krum Attack | 与 Trim Attack 同源同类（Fang et al. 2020 AGR-specific 家族），只需一个代表。Trim Attack 因 Trimmed Mean 是我们的基线而优先 |
| Min-Sum | 与 Min-Max 同源同类（Shejwalkar & Houmansadr 2021 AGR-agnostic 家族），Min-Max 更强，保留强者即可 |
| MPAF | sybil + 放大，但放大测试与 Scaling Attack 重叠，sybil 场景在 10 客户端设定下不够典型 |
| BadNets | 最基础像素后门，被 Scaling Attack（同样是像素后门 + 放大）完全覆盖 |
| Constrain-and-Scale | Scaling Attack 的方向对齐变体，对 VeriFL Phase 2 的挑战方式相同（核心仍是模长放大） |
| Edge-case | 语义后门（自然图片触发），有趣但仅 FLAME 测试，基线覆盖不足 |
| Gaussian Noise | 过于简单，VeriFL Phase 1 即可轻松应对，浪费实验篇幅 |
| Sign-flipping | 与 Label Flipping 在效果上类似（方向偏离），仅 FLAD 测试 |
| ALIE (LIE) | 虽对 Phase 1 有趣（刚好低于统计检测阈值），但不在 5 篇核心基线的实验中，无法形成公平对比 |

---

## 4. 基线对比公平性检查

| 基线 | 可直接对比的攻击 | 数量 |
|------|---------------|------|
| FLTrust | Label Flipping, Trim Attack, Scaling | 3 |
| FLAME | Scaling, DBA | 2 |
| FLAD | Label Flipping, Scaling, DBA | 3 |
| Fang'25 | Label Flipping, Trim Attack, Min-Max, Scaling, DBA, Neurotoxin | 6 |
| FilterFL | Scaling, DBA, Neurotoxin | 3 |

> FLTrust / FLAD / Fang'25 / FilterFL 均有 ≥ 3 个重叠攻击，对比充分。FLAME 有 2 个（Scaling + DBA），因 FLAME 原文的其余攻击（Constrain-and-Scale, Edge-case）都是 Scaling 家族变体或小众攻击，2 个覆盖足以构成公平比较。

---

## 5. 执行建议

- **实现顺序**: ① Scaling Attack → ② Label Flipping → ③ DBA → ④ Neurotoxin → ⑤ Min-Max → ⑥ Trim Attack → ⑦ Adaptive Attack
  - 原因：先实现后门基础设施（Scaling），再扩展到分布式（DBA）和持久（Neurotoxin）；untargeted 相对独立可穿插；Adaptive 最后做（需要先理解框架）
- **恶意比例**: 默认 30%（3/10，与当前代码配置一致），扩展测 20%/40%/50%
- **数据集**: CIFAR-10（主实验） + MNIST（辅助验证泛化性）
- **Non-IID**: Dirichlet $\alpha \in \{0.1, 0.5\}$
- **每攻击记录**: MTA (Main Task Accuracy), ASR (仅 targeted), 收敛曲线
- **模型**: ResNet-20（与当前代码 `context/algorithm.md §15.2` 一致）
