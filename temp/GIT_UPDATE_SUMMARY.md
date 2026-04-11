# Git 更新全量总结汇报

> **扫描范围**：`3ff689a4..dae657109`（25 commits）  
> **扫描日期**：2026-04-10  
> **当前 HEAD**：`dae6571095d0`（`master` = `origin/master`）  
> **变更规模**：581 files changed, 61,911 insertions(+), 403 deletions(-)

---

## 一、提交时间线（按时间正序）

按里程碑阶段划分，25 个提交可归纳为以下 6 个阶段：

### 阶段 A：Phase 2 基础建设（6 commits）
| Commit | 摘要 |
|--------|------|
| `98891acac` | chore: phase 2 & prepare for gpu test |
| `e3385c0a0` | chore: phase 2 gpu foundation |
| `726f70200` | chore: more test result |
| `4264f16a6` | chore: REMOTE |
| `f3506861c` | chore: update |
| `aadbf21f4` | chore: script for gpu test |

**要点**：搭建 GPU 实验环境、远程执行框架、初步跑通 Phase 2 实验。

### 阶段 B：M1 基线闭环（7 commits）
| Commit | 摘要 |
|--------|------|
| `594417bda` | fix: arg error |
| `c7576f133` | chore: remove remote server |
| `dff891741` | chore: phase 2 v1 |
| `33b295146` | chore: sync GPU experiment results (M1/M2/M3) |
| `db0d28c4b` | chore: remove config |
| `9fbc6a4d8` | chore: m1 e5 doc |
| `83fe4cd28` | **M1 闭环实验：基线可信达标** (E=1, server_lr=1.0, wd=1e-4/0) |

**要点**：M1 基线（FedAvg 无攻击无防御）全面闭环，确认 E=1 参数配置下 CIFAR-10 和 MNIST 基线精度可信。

### 阶段 C：M1.5 Label Flipping 实验（4 commits）
| Commit | 摘要 |
|--------|------|
| `0a7be5732` | chore: m1 new req |
| `2d4bac010` | chore: doc dir |
| `a6987bb17` | chore: before m1.5 |
| `420ff28d3` | **M1.5: add experiment report, 24 JSONL results, 24 YAML configs** |

**要点**：完成 24 组 Label Flipping 实验（N=10, PMR=30%, CIFAR-10 + MNIST, 4 alpha × 3 seeds），产出结构化 JSONL 结果和实验报告。

### 阶段 D：M2 攻击扩展 — Scaling Attack 修复与初始实验（3 commits）
| Commit | 摘要 |
|--------|------|
| `02a90c96c` | chore: m2 |
| `bde23de62` | chore: m2 sa |
| `42d8d7f9d` | chore: m2 error |

**要点**：开始 M2 Scaling Attack (Model Replacement) 实验，遇到若干 bug 并修复。

### 阶段 E：P1/P1.5 正式实验 — N=50 规模（3 commits）
| Commit | 摘要 |
|--------|------|
| `402f47f98` | **feat: add Phase 1 experiment results** (P1 LF + SA, defense=none FedAvg) |
| `ca89f680e` | **P1.5 Scaling Attack: auto-gamma + single-round attack** (35 experiments) |
| `3c04b9820` | P1.5: add AC analysis script for experiment results |

**要点**：完成 P1 阶段全部攻击实验（N=10 Label Flipping + Scaling Attack），以及 P1.5 阶段 N=50 规模的 Scaling Attack 正式实验（35 组），引入 auto-gamma 和 single-round 攻击模式。

### 阶段 F：架构重构 — 可插拔防御 + VeriFL 重命名（2 commits）
| Commit | 摘要 |
|--------|------|
| `225889eb8` | **feat(shieldfl): N=50 formal experiments — code, results & analysis report** |
| `6772defd0` | **refactor: pluggable defense registry + VeriFL → v16 rename** |
| `dae657109` | chore: fmt |

**要点**：FedML 防御框架核心重构 + VeriFL 版本化命名 + N=50 正式实验报告。

---

## 二、核心代码变更详解

### 2.1 FedML 核心框架层

#### 2.1.1 `fedml_defender.py` — 可插拔防御注册表（最大架构变更）

**改动前**：17 个 `if-elif` 硬编码分支选择防御算法，阶段判断 (`is_defense_before_aggregation()` 等) 也是硬编码列表。

**改动后**：
- 引入 `_DefenseRegistry` 模块级单例，所有 16 个内置防御通过 `register_lazy()` 延迟注册
- 每个防御声明所参与的 hook 阶段（`before_aggregation` / `on_aggregation` / `after_aggregation`）
- `FedMLDefender.init()` 通过注册表查找 + 实例化，开头强制 state reset 防止跨实验泄漏
- 新增公开 API：
  - `FedMLDefender.register_defense(type, cls, phases)` — 外部零侵入注册自定义防御
  - `FedMLDefender.available_defenses()` — 查询所有可用防御
- `is_defense_*_aggregation()` 方法改为查询 `_phases` 元数据，消除硬编码

**影响**：M3 阶段（防御实验）的基础设施彻底完成。VeriFL 或任何自定义防御现在可以通过一行 `register_defense()` 接入，无需修改框架代码。

#### 2.1.2 `client_trainer.py` — 数据投毒隔离

- `update_dataset()`: 只对训练数据投毒，测试数据显式保持干净
- `is_to_poison_data()` 新增 `client_id` 参数，支持确定性的按客户端投毒决策

#### 2.1.3 `fedml_attacker.py` — 攻击管理器

- `is_to_poison_data()` 转发 `client_id` 和 `round_idx` 给具体攻击实现

#### 2.1.4 `fedml/__init__.py` — MPI cpu_transfer 修复

- 非 TRPC 后端不再强制覆盖 `cpu_transfer=False`，改为 `getattr(args, "cpu_transfer", False)` 尊重用户配置

#### 2.1.5 `cross_silo/server/fedml_aggregator.py` — GPU OOM 修复

- 聚合时增加设备管理逻辑，防止 N=50 规模下 GPU 显存溢出

### 2.2 攻击子系统

#### 2.2.1 `label_flipping_attack.py` — 全面重写

| 修复项 | 改动前 | 改动后 |
|--------|--------|--------|
| D1: 标签翻转重叠 | 逐标签 for 循环可双重覆写 | mapping dict + 掩码赋值一次性完成 |
| D2: 恶意客户端不固定 | 每轮基于 `np.random.seed(counter)` 随机选择 | 固定集合，init 时用隔离 RNG 确定 |
| D3: 标签 dtype 丢失 | 用 `torch.Tensor` 累加（float） | 用 `torch.LongTensor` 保持整数 |
| D4: DataLoader shuffle 丢失 | `shuffle=False`（默认） | 显式 `shuffle=True` |
| D5: 测试数据被污染 | train 和 test 都投毒 | 仅 train 投毒 |
| D6: 全局 RNG 污染 | `np.random.seed()` 全局设置 | `np.random.default_rng()` 隔离 RNG |
| D7: 轮次跟踪错误 | `counter / client_num_per_round` | 直接接受 `round_idx` 或自增 |
| W8: 审计日志 | 无 | init 和每轮都有结构化日志 |

#### 2.2.2 `model_replacement_backdoor_attack.py` — 重大增强

| 改动 | 详情 |
|------|------|
| `scale_gamma` | 支持 `"auto"` (= N/K 自动计算) 和固定值，取代旧的 `participant_num` 硬编码 |
| `malicious_client_ids` | 固定为 `[0, 1, ..., K-1]`，不再每轮随机 |
| `attack_training_rounds` | 支持指定攻击轮次列表（single-round attack） |
| Full participation 断言 | `client_num_per_round == client_num_in_total` |
| In-place scaling | 不再 pop+insert，直接对恶意客户端模型原地缩放 |
| `should_scale_param()` | 排除 `num_batches_tracked`，但保留 `running_mean/var` |
| `self.last_gamma` | 暴露每轮实际 gamma 值供结构化指标记录 |

#### 2.2.3 `common/utils.py`

- 新增 `should_scale_param(k)` — BN 参数缩放判断
- `get_malicious_client_id_list()` 改用隔离 RNG
- `replace_original_class_with_target_class()` 改为 mapping dict 一次性赋值，修复双重覆写 bug

### 2.3 ShieldFL 实验项目层

#### 2.3.1 入口重构

| 改动前 | 改动后 |
|--------|--------|
| `VeriFLAggregator` | `ShieldFLAggregator`（新建，统一入口） |
| `VeriFLTrainer` | `VeriFLv16Trainer` |
| `verifl_aggregator.py` | `verifl_v16_aggregator.py`（重命名） |
| 旧 `baseline_aggregator.py` | 删除，功能并入 `ShieldFLAggregator` |

**`ShieldFLAggregator`** 继承 `ServerAggregator`，整合了：
- 标准 FedAvg 聚合通路（继承框架 hook chain）
- ASR 评估（自动触发器归一化）
- `MetricsCollector` 结构化 JSONL 指标记录
- `gamma_actual` 追踪（从 attacker 实例读取 `last_gamma`）

#### 2.3.2 `VeriFLv16Trainer` — 客户端训练器

- 支持 Model Replacement 本地后门注入：`attack_training_rounds`, `backdoor_per_batch`, trigger 自动归一化
- 隔离 RNG 保证可复现性
- `_NORM_PARAMS` 匹配 `data_loader.py` 的归一化参数

#### 2.3.3 `data_loader.py` — 数据加载增强

- **新增 MNIST 支持**：完整的 MNIST 加载 + 归一化管道
- **Dirichlet 分区改进**：
  - 移除 boolean cap（D-12 决策：对齐社区标准 Dirichlet）
  - 增加 `min_samples=10` + `max_retries=100` 重试保护
  - 详细分区统计日志（AC-C-4 验证用）
- **分层均衡采样**：新增 `_stratified_balanced_sample()` 用于服务端验证集
- **去除服务端验证集数据增强**：统一不使用 RandomCrop/RandomFlip（与学术标准文献对齐）

#### 2.3.4 评估模块 (`eval/`)

- **`metrics.py`**：结构化 JSONL 指标采集器
  - 文件名编码：`model_dataset_aggregator_attack_defense_alpha_pmr_gamma_seed`
  - 字段：`round, test_accuracy, test_loss, asr, agg_time, gamma_actual, timestamp` + 完整 run metadata
- **`asr.py`**：后门 ASR 评估
  - 自动触发器归一化（pixel space → normalized space，按 dataset 自动转换）
  - 仅对 `label != target_label` 的样本注入触发器

#### 2.3.5 模型

- 新增 `lenet5.py` — LeNet5 for MNIST
- 新增 `resnet18.py` — ResNet18 for CIFAR-10

#### 2.3.6 实验脚本

| 脚本 | 用途 |
|------|------|
| `run_experiment.sh` | 核心实验启动器，自动生成 YAML + MPI 运行 |
| `batch_n50.sh` | N=50 批量实验编排脚本 |
| `gpu_wrapper.sh` | GPU 资源管理包装器 |
| `run_m2_lf_gpu.sh` | M2 Label Flipping GPU 实验矩阵 |
| `run_m2_scaling_full.sh` | M2 Scaling Attack 完整实验矩阵 |
| `run_p1_lf_gpu.sh` / `run_p1_sa_gpu.sh` | P1 阶段实验脚本 |
| `run_p1.5_sa_gpu.sh` | P1.5 Scaling Attack GPU 实验 |
| `analyze_p1.5_results.py` | P1.5 实验结果 AC 分析 |
| `test_lf_correctness.py` | Label Flipping 代码正确性测试 (AC-1 ~ AC-6) |

#### 2.3.7 测试

| 测试 | 覆盖 |
|------|------|
| `smoke_test_pluggable_defense.py` | 可插拔防御注册表单元测试 (T1-T6) |
| `smoke_e2e.sh` | 端到端 GPU 冒烟测试 (T7) |
| `test_scaling_correctness.py` | Scaling Attack 代码正确性测试 |

---

## 三、实验结果清单

### 3.1 已入库结果（Git 已 tracked）

| 阶段 | 攻击 | 数据集 | N | PMR | 组数 | JSONL |
|------|------|--------|---|-----|------|-------|
| M1 基线 | none | CIFAR-10 + MNIST | 10 | 0% | 24 | ✅ |
| M1.5 LF | label_flipping | CIFAR-10 + MNIST | 10 | 30% | 24 | ✅ |
| P1 LF | label_flipping | CIFAR-10 + MNIST | 10 | 30% | 24 | ✅ |
| P1 SA | model_replacement | CIFAR-10 + MNIST | 10 | 30% | 24+ | ✅ |
| P1.5 SA (N=50) | model_replacement | CIFAR-10 + MNIST | 50 | 10-20% | 28 | ✅ |

合计已入库 JSONL 文件：**156 个**，配套 YAML 配置：**106 个**。

### 3.2 N=50 正式实验核心发现

**Scaling Attack (Model Replacement) — N=50, PMR=20%, auto-gamma:**

| 数据集 | α=0.1 | α=0.5 | α=100 |
|--------|-------|-------|-------|
| **CIFAR-10 Median Acc** | 10.0% (NaN 2/3) | 39.4% | 73.8% |
| **CIFAR-10 Mean ASR** | 99.4% | 96.6% | 98.9% |
| **MNIST Median Acc** | 96.9% | 97.0% | 88.4% |
| **MNIST Mean ASR** | 99.98% | 99.99% | 99.14% |

**AC 验收**：5/7 PASS，2 FAIL（AC-A-3 NaN 数值稳定性、AC-A-7 因果性 ΔASR）均为学术设计问题而非代码缺陷。

---

## 四、文档体系

### 4.1 根目录文档（32 个 .md）

| 类别 | 文档 | 内容 |
|------|------|------|
| **基线** | M1_EXPERIMENT_REPORT, M1_EXPERIMENT_PARAMS, M1_ACADEMIC_CONSISTENCY | M1 基线参数/结果/学术一致性验证 |
| **LF 实验** | M1.5_EXPERIMENT_REPORT, M2_LF_EXPERIMENT_REPORT | Label Flipping 实验报告 |
| **SA 修复** | M2_SA_FIX_PHASE0/1/1_ERROR/1_ERROR2/1_ERROR_FIX/1_ERROR2_FIX | Scaling Attack 6 轮 debug 全记录 |
| **N=50 正式** | N50_EXPERIMENT_REPORT, N50_EXPERIMENT_GUIDE | N=50 正式实验报告和执行指南 |
| **架构** | PLUGGABLE_DEFENSE_CHANGELOG | 可插拔防御注册表变更文档 |
| **规划** | PHASE2, PHASE2_GPU, PHASE3 | Phase 2/3 推进方案 |
| **学术** | M1.5学术标准文档, M1 闭环学术需求 | 学术标准和需求文档 |
| **Scaling 定稿** | Scaling_实施定稿 | Scaling Attack 完整实施规格 |

### 4.2 `重生推进方案_实施/` 目录（12 个文档）

包含完整的研究推进计划：重生推进方案.md、VeriFL 升级方案、ShieldFL 基础设施审计、威胁模型、攻击清单、算法文档、LF/Scaling 实施规格、失败根因分析、LF 投毒失效调查等。

### 4.3 `LF 复现与验收/` 目录（4 个文档）

Label Flipping 专项：LF 实施规格、攻击清单、威胁模型，以及两篇相关参考文献笔记（Fang 2020, Fang 2025）。

---

## 五、架构现状总结

```
┌─── FedML Core Framework ─────────────────────────────────────┐
│                                                              │
│  FedMLAttacker (单例)                                        │
│    ├─ LabelFlippingAttack     ← 全面重写 (D1~D7)            │
│    └─ ModelReplacementAttack  ← 增强 (auto-γ, fixed IDs)    │
│                                                              │
│  FedMLDefender (单例)                                        │
│    └─ _DefenseRegistry        ← 新! 16 内置 + 可扩展        │
│                                                              │
│  ServerAggregator                                            │
│    └─ on_before / aggregate / on_after hooks                 │
│                                                              │
│  ClientTrainer                                               │
│    └─ update_dataset → 仅 poisoned train data                │
│                                                              │
└──────────────────────────────────────────────────────────────┘

┌─── ShieldFL Experiment Host ─────────────────────────────────┐
│                                                              │
│  main_fedml_shieldfl.py (入口)                               │
│    ├─ VeriFLv16Trainer      (client trainer + backdoor)      │
│    ├─ ShieldFLAggregator    (server, FedAvg path + metrics)  │
│    └─ VeriFLv16Aggregator   (VeriFL path, GA+anchor+BN)     │
│                                                              │
│  eval/                                                       │
│    ├─ metrics.py  (JSONL structured metrics)                 │
│    └─ asr.py      (trigger-based ASR evaluation)             │
│                                                              │
│  data/                                                       │
│    └─ data_loader.py  (CIFAR-10 + MNIST, Dirichlet, N≤50)   │
│                                                              │
│  model/                                                      │
│    ├─ ResNet18, ResNet20, SimpleCNN                          │
│    └─ LeNet5                                                 │
│                                                              │
│  scripts/  (30+ 实验/分析脚本)                               │
│  tests/    (pluggable defense + scaling correctness)         │
│  results/  (156 JSONL + 106 YAML configs)                    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 六、关键风险与建议

### 6.1 已解决的风险

| 风险 | 解决方案 | 状态 |
|------|----------|------|
| FedAvg 基线语义不纯 | 拆分 ShieldFLAggregator 作为标准 FedAvg 通路 | ✅ |
| 标签翻转测试数据泄漏 | `client_trainer.py` 只 poisoned train data | ✅ |
| 恶意客户端集合不固定 | init 时用隔离 RNG 确定固定集合 | ✅ |
| 防御类型添加需改核心代码 | 可插拔注册表模式 | ✅ |
| 跨实验状态泄漏 | `init()` 强制 state reset | ✅ |
| GPU OOM (N=50) | `fedml_aggregator.py` 设备管理修复 | ✅ |

### 6.2 当前待关注项

| 项目 | 详情 | 优先级 |
|------|------|--------|
| AC-A-3 NaN | CIFAR-10 α=0.1 + Scaling Attack 在 N=50 下 2/3 seed 出现 loss NaN | 中 — 学术设计层面，非代码 bug |
| AC-A-7 因果性 | γ=1 控制组 ASR 仍达 ~95%，scaling 边际贡献不足 30pp | 高 — 影响论文叙事 |
| `gamma_actual` JSONL 字段 | ShieldFLAggregator 已读取 `last_gamma`，但部分旧结果该字段为 None | 低 — 不影响功能 |
| M3 防御实验 | 注册表已就绪，但尚未开始 M3 防御实验 | 高 — 下一阶段重点 |
| VeriFL v16→v18f 升级 | 方案文档已有，代码尚未实施 | 中 — 取决于 M3 排期 |
| 部分根目录文档冗余 | 根目录已有 32 个 .md，M2_SA_FIX 系列有 6 个 debug 文档 | 低 — 建议后续整理归档 |

---

## 七、数据完整性快照

| 指标 | 数值 |
|------|------|
| 已入库 JSONL 结果 | 156 |
| 已入库 YAML 配置 | 106 |
| FedML 核心文件修改 | 8 |
| ShieldFL 项目文件修改/新增 | ~35 |
| 文档文件（根 + 子目录） | ~50 |
| 实验脚本 | 30+ |
| 测试用例 | 3 (smoke + scaling + pluggable defense) |
| 工作区状态 | **干净**（无未提交变更） |
