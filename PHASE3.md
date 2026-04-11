# ShieldFL → FedML Phase 3 实施清单

> 本文基于以下上游文档推导：
> - **MOVE.md**：总体迁移方案与验收标准
> - **PHASE1.md / PHASE2.md / PHASE2_GPU.md**：已交付的三个阶段实施记录
> - **PHASE2_GPU_RESULT.md**：Phase 2 GPU 实际执行偏差与当前状态
> - **学术需求.md**：M1–M4 学术门槛
>
> Phase 3 的关键词：**收尾、统一、固化——将已跑通的攻防全链路和 GPU 实验基础设施打磨为"可随需求自由调参并产出可复核学术结果"的完整实验框架。**

---

## 1. Phase 3 的目标

Phase 3 必须同时满足三件事：

1. **实验完整性**：M1/M2/M3 在两条任务线上的实验矩阵全量完成（或以明确策略完成有代表性的子集），M4 结果固化交付包就绪。
2. **框架可用性**：任何人拿到本项目后，只需修改 `run_experiment.sh` 参数即可覆盖任意 model × dataset × attack × defense × PMR × α × seed × epochs × rounds 组合，无需改代码。
3. **代码整洁性**：清理历史遗留的冗余配置、死代码、过时结果文件和文档歧义，使仓库状态与实际运行能力一致。

Phase 3 结束时的状态应为：

> **仓库是一个干净、自文档化、可复现的联邦学习攻防实验框架：配置即实验，脚本即矩阵，结果即验收。**

---

## 2. 前置条件

### 2.1 Phase 2 GPU 已交付 / 在途能力

| 能力 | 状态 | 说明 |
|---|---|---|
| FedML cross-silo + MPI + GPU 全链路 | ✅ | 3 GPU 并行稳定运行 |
| FedAvg + VeriFL 双聚合器 | ✅ | 通过 `--aggregator` 切换 |
| 3 类 FedML 内置攻击 | ✅ | byzantine / label_flipping / model_replacement |
| 4 类 FedML 内置防御 | ✅ | rfa / krum / trimmed_mean / cclip |
| ASR 评估 | ✅ | eval/asr.py |
| 结构化指标采集 | ✅ | eval/metrics.py → JSONL |
| 实验编排脚本 | ✅ | run_experiment.sh 动态生成 YAML |
| M1/M2/M3 GPU 实验 | 🔄 运行中 | M1 epochs=5 进行中，M2/M3 优先子集进行中 |
| 结果汇总脚本 | ✅ 就绪 | summarize_results.py |

### 2.2 Phase 3 启动条件

Phase 3 中的大部分清理和框架打磨工作可以与 Phase 2 GPU 实验运行并行推进。以下条件需在 Phase 3 最终验收前满足：

- M1 epochs=5 至少完成一组 α 值（如 α=0.5 × 3 seeds），确认学术门槛可达
- M2 优先子集至少完成 byzantine + label_flipping 的 CIFAR10 部分
- M3 优先子集至少完成 krum + byzantine 的 CIFAR10 部分

---

## 3. Phase 3 实施步骤

### Step 1：统一 local epochs 并确定最终实验参数

**问题**：PHASE2_GPU_RESULT.md 记录 M1 从 epochs=1 调整到 epochs=5，但 M2/M3 优先子集仍使用 epochs=1。学术需求.md §3.2 明确要求"本地训练预算必须跨所有攻防组合保持一致"。

**操作**：

1. 等 M1 epochs=5 首批结果（至少 α=0.5 × 3 seeds）出齐后，确认是否达标
2. 若达标：冻结 epochs=5 为全局统一值
3. 若仍不达标：可考虑 epochs=10，但必须记录校准变更（原值/新值/原因/证据实验）
4. 确定后修改所有 M2/M3 脚本的 epochs 参数至统一值
5. M2/M3 优先子集中 epochs=1 的旧结果归档或标记为"预热实验"，不作为学术验收依据

**最终确定的参数应写入一份 `EXPERIMENT_PARAMS.md`**：

```markdown
# 冻结实验参数
| 参数 | 值 | 适用范围 |
|---|---|---|
| local epochs | 5（待确认） | 全部 M1/M2/M3 |
| batch_size | 64 | 全部 |
| client_optimizer | SGD | 全部 |
| learning_rate | 0.01 | 全部 |
| momentum | 0.9 | 全部 |
| client_num_in_total | 10 | 全部 |
| client_num_per_round | 10 | 全部 |
| comm_round (CIFAR10) | 100 | M1/M2/M3 |
| comm_round (MNIST) | 50 | M1/M2/M3 |
| val_per_class | 50 | 全部 |
| trust_per_class | 50 | 全部 |
| pop_size | 15 | VeriFL |
| generations | 10 | VeriFL |
| server_momentum | 0.9 | VeriFL |
| server_lr | 0.3 | VeriFL |
| lambda_reg | 0.01 | VeriFL |
| seeds | {0, 1, 2} | 全部 |
| α (Dirichlet) | {0.1, 0.3, 0.5, 100} | 全部 |
| PMR | {0.1, 0.2, 0.3, 0.4} | M2/M3 |
```

**验证点**：
- 最终 epochs 值确定并冻结
- 所有编排脚本使用同一 epochs
- 不存在 M1 用 epochs=5 而 M2 用 epochs=1 的不一致

---

### Step 2：补齐 M2/M3 MNIST 任务线

**问题**：PHASE2_GPU_RESULT.md 显示 MNIST 子集"尚未开始"。学术需求.md 要求两条任务线均覆盖。

**操作**：

1. 确认 MNIST 数据集在 GPU 服务器上可用（data 目录已有 MNIST/raw/）
2. 在 M1 完成后，运行 MNIST 的 M1 基线实验（LeNet5 + MNIST × 4α × 3 seeds = 12 次）
3. M2 MNIST 优先子集：byzantine + label_flipping + model_replacement × PMR=0.2 × α∈{0.1, 0.5} × 3 seeds = 18 次
4. M3 MNIST 优先子集：krum + trimmed_mean × byzantine × PMR=0.2 × α∈{0.1, 0.5} × 3 seeds = 12 次
5. VeriFL + MNIST：byzantine × PMR=0.2 × α∈{0.1, 0.5} × 3 seeds = 6 次

**注意**：MNIST 的学术门槛（MA ≥ 97%）比 CIFAR10 宽松得多，LeNet5 规模很小，每次实验应在数分钟内完成。

**验证点**：
- MNIST M1 达到 MA ≥ 97%
- MNIST M2 攻击效果可观测（MA 明显下降或 ASR 明显上升）

---

### Step 3：清理冗余配置文件

**操作**：将以下文件移入 `config/legacy/` 子目录（不删除，保留历史可追溯性）：

```
config/fedml_config_cpu_observable.yaml      → config/legacy/
config/fedml_config_cpu_fidelity.yaml        → config/legacy/
config/fedml_config_m1_cifar10_cpu.yaml      → config/legacy/
config/fedml_config_m1_mnist_cpu.yaml        → config/legacy/
config/fedml_config_m2_byzantine_cpu.yaml    → config/legacy/
config/fedml_config_m3_quick_cpu.yaml        → config/legacy/
config/fedml_config_m3_verifl_vs_attack_cpu.yaml → config/legacy/
config/fedml_config_m1_cifar10_gpu.yaml      → config/legacy/
config/fedml_config_m1_mnist_gpu.yaml        → config/legacy/
config/fedml_config_verifl_cifar10_gpu.yaml  → config/legacy/
```

保留：
- `config/gpu_mapping.yaml`（运行时使用）

**理由**：run_experiment.sh 动态生成 YAML 是唯一的实验入口，静态 YAML 文件既不被使用也容易误导。移入 legacy/ 后保留历史参考价值。

**验证点**：
- run_experiment.sh 仍能正常动态生成 YAML 并运行
- legacy/ 目录包含移入的文件

---

### Step 4：清理冗余代码

#### 4.1 移除 BaselineAggregator 中的 Bulyan 死代码

**文件**：`trainer/baseline_aggregator.py`

**操作**：移除 `_aggregate_bulyan()` 方法及 `aggregate()` 中的 `if defense_type == "bulyan"` 分支。

**理由**：PHASE2_GPU_RESULT.md 确认 Bulyan 不在 FedML 已注册防御中，已从实验矩阵移除。该代码路径永远不会被触发。

#### 4.2 清理结果文件

**操作**：

```
results/old_epochs1/           → 保留但在 README 中标记为"已归档的预热实验"
results/alignment_cpu/         → 移入 results/legacy/
results/alignment_gpu/         → 移入 results/legacy/
results/ 下的 SimpleCNN / ResNet20 JSONL → 移入 results/legacy/
```

正式学术实验结果应统一放在 `results/` 根目录下，命名规范不变。

#### 4.3 CPU 编排脚本标记

**操作**：在 `scripts/run_m1_baseline_cpu.sh`、run_m2_attacks_cpu.sh、`run_m3_defense_cpu.sh` 文件头部添加注释：

```bash
# [LEGACY] Phase 2 CPU smoke test 脚本。
# 正式 GPU 实验请使用 run_m1_baseline_gpu.sh / run_m2_priority_gpu.sh / run_m3_priority_gpu.sh
```

不删除，因为可能仍用于无 GPU 环境的调试。

**验证点**：
- `_aggregate_bulyan()` 已移除
- 结果目录结构清晰：正式结果在 `results/`，历史产物在 `results/legacy/`

---

### Step 5：完善 `run_experiment.sh` 的文档和健壮性

**问题**：`run_experiment.sh` 是整个框架的核心入口，但缺乏使用说明和某些边界处理。

**操作**：

1. 在脚本顶部补充完整用法说明，列出所有参数及其默认值：

```bash
# 用法示例：
#   # M1：FedAvg 基线
#   bash scripts/run_experiment.sh \
#     --model ResNet18 --dataset cifar10 --aggregator fedavg \
#     --attack none --defense none \
#     --alpha 0.5 --seed 0 --rounds 100 --clients 10 --epochs 5 \
#     --gpu --gpu_id 0 --runtime single-gpu-deterministic
#
#   # M2：byzantine 攻击
#   bash scripts/run_experiment.sh \
#     --model ResNet18 --dataset cifar10 --aggregator fedavg \
#     --attack byzantine --defense none --pmr 0.2 \
#     --alpha 0.5 --seed 0 --rounds 100 --clients 10 --epochs 5 \
#     --gpu --gpu_id 1 --runtime single-gpu-deterministic
#
#   # M3：VeriFL 防御
#   bash scripts/run_experiment.sh \
#     --model ResNet18 --dataset cifar10 --aggregator verifl \
#     --attack byzantine --defense none --pmr 0.2 \
#     --alpha 0.5 --seed 0 --rounds 100 --clients 10 --epochs 5 \
#     --gpu --gpu_id 0 --runtime single-gpu-deterministic
#
# 全部参数：
#   --model MODEL          模型名称 (ResNet18|LeNet5|SimpleCNN|ResNet20)
#   --dataset DATASET      数据集 (cifar10|mnist)
#   --attack ATTACK        攻击类型 (none|byzantine|label_flipping|model_replacement)
#   --defense DEFENSE      防御类型 (none|rfa|krum|trimmed_mean|cclip)
#   --aggregator AGG       聚合器 (fedavg|verifl)
#   --pmr PMR              恶意客户端比例 (0.0-1.0)
#   --alpha ALPHA          Dirichlet α (0.1|0.3|0.5|100 等)
#   --seed SEED            随机种子
#   --rounds ROUNDS        通信轮数
#   --clients CLIENTS      客户端总数
#   --epochs EPOCHS        本地训练 epoch 数
#   --batch_size BATCH     批大小
#   --gpu                  启用 GPU
#   --gpu_id ID            指定 GPU 编号 (默认 0)
#   --runtime MODE         运行模式 (cpu-deterministic|single-gpu-deterministic|single-gpu-fast)
```

2. 添加参数合法性检查：VeriFL 聚合器 + 内置防御不允许同时启用（双重防御隔离）：

```bash
if [[ "$AGGREGATOR" == "verifl" && "$DEFENSE" != "none" ]]; then
    echo "ERROR: VeriFL aggregator cannot be combined with FedML built-in defense (double defense)."
    echo "Use --aggregator fedavg with --defense, or --aggregator verifl with --defense none."
    exit 1
fi
```

3. 确保动态生成的 YAML 中 epochs 取自命令行参数而非硬编码。

**验证点**：
- `--help` 或脚本头部注释足以说明所有参数
- VeriFL + 内置防御的非法组合被拒绝
- epochs 由命令行参数控制，不存在硬编码

---

### Step 6：补做 Aggregation Time 基准与 CPU/GPU 对齐（可选但推荐）

**问题**：PHASE2_GPU_RESULT.md 跳过了 CPU vs GPU 数值对齐和 Aggregation Time 基准。MOVE.md §11.3 要求"在单 GPU 下同 seed 可复现主要指标"。

**操作**：

1. 在 M1 epochs=5 完成后，用同一 seed/α/epochs 跑一轮 CPU 和 GPU 的 FedAvg 5 轮短实验：

```bash
# CPU baseline
bash scripts/run_experiment.sh \
  --model ResNet18 --dataset cifar10 --aggregator fedavg \
  --attack none --defense none --alpha 0.5 --seed 0 \
  --rounds 5 --clients 3 --epochs 5 --batch_size 64

# GPU run
bash scripts/run_experiment.sh \
  --model ResNet18 --dataset cifar10 --aggregator fedavg \
  --attack none --defense none --alpha 0.5 --seed 0 \
  --rounds 5 --clients 3 --epochs 5 --batch_size 64 \
  --gpu --runtime single-gpu-deterministic
```

2. 用 compare_cpu_gpu_metrics.py 对比前 5 轮 MA 差异
3. 对 VeriFL 聚合器做同样对比（容差放宽到 1.0%）
4. 记录对齐结果

**Aggregation Time 基准**：

```bash
# 同配置的 VeriFL CPU vs GPU
bash scripts/run_experiment.sh \
  --model ResNet18 --dataset cifar10 --aggregator verifl \
  --attack none --defense none --alpha 0.5 --seed 0 \
  --rounds 5 --clients 5 --epochs 5 --batch_size 64

bash scripts/run_experiment.sh \
  --model ResNet18 --dataset cifar10 --aggregator verifl \
  --attack none --defense none --alpha 0.5 --seed 0 \
  --rounds 5 --clients 5 --epochs 5 --batch_size 64 \
  --gpu --runtime single-gpu-deterministic
```

从两份 JSONL 中提取 `agg_time` 列对比。

**验证点**：
- FedAvg 前 5 轮 |Δ MA| ≤ 0.5%
- VeriFL 前 3 轮 |Δ MA| ≤ 1.0%
- GPU agg_time < CPU agg_time（VeriFL）
- 硬件上下文已记录在 JSONL 中

---

### Step 7：决定实验覆盖策略并补跑

**问题**：全矩阵规模巨大（M2 全矩阵 288 次，M3 全矩阵 1728 次），需明确最终覆盖策略。

**推荐策略：分层覆盖**

#### 必做层（学术门槛验收）

以下实验必须完成，因为学术需求.md 的门槛判定直接依赖这些结果：

**M1**（全量 24 次）：4α × 3 seeds × 2 tasks

**M2 核心子集**（108 次）：
- 3 attacks × PMR=0.2 × 4α × 3 seeds × 2 tasks = 72 次（覆盖所有 α）
- 3 attacks × PMR∈{0.1, 0.3, 0.4} × α∈{0.1, 0.5} × 3 seeds × 1 task(CIFAR10) = 54 次（覆盖 PMR 敏感度）
- 合计去重后约 108 次

**M3 核心子集**（132 次）：
- 4 defenses × 3 attacks × PMR=0.2 × α∈{0.1, 0.5} × 3 seeds × 1 task(CIFAR10) = 72 次
- VeriFL × 3 attacks × PMR=0.2 × α∈{0.1, 0.5} × 3 seeds × 1 task(CIFAR10) = 18 次
- MNIST 核心：(4 defenses + VeriFL) × byzantine × PMR=0.2 × α=0.5 × 3 seeds = 15 次
- 无攻击防御保真度：4 defenses × none × α∈{0.1, 0.5} × 3 seeds × 1 task = 24 次
- 合计约 129 次

#### 推荐扩展层（如时间允许）

- M2：扩展到全 4 PMR × 全 4 α × 2 tasks = 288 次
- M3：扩展到全 4 PMR × 全 4 α × 3 attacks × 2 tasks

#### 实施方式

编写一组"最终实验"脚本，区别于"优先子集"脚本：

```bash
scripts/run_m2_final_gpu.sh    # M2 核心子集
scripts/run_m3_final_gpu.sh    # M3 核心子集
```

脚本内应包含 `check_done()` 逻辑，自动跳过已有结果。

**验证点**：
- M1 全量 24 次完成
- M2 核心子集 108 次完成（或以明确策略确定的子集）
- M3 核心子集 132 次完成（或以明确策略确定的子集）

---

### Step 8：运行 summarize_results.py 产出 M4 汇总表

**操作**：

1. 确认 summarize_results.py 能正确解析所有 JSONL 文件名和内容
2. 运行，产出 Markdown 格式汇总表
3. 检查结果是否满足学术门槛

```bash
cd python/examples/federate/prebuilt_jobs/shieldfl/
python scripts/summarize_results.py --results_dir ./results --format markdown > RESULTS_SUMMARY.md
```

**汇总表应包含**：

- M1 基线表：model × dataset × α → MA (mean ± std)
- M2 攻击效果表：model × dataset × attack × PMR × α → MA (mean ± std) + ASR (mean ± std)
- M3 防御对照表：model × dataset × defense × attack × PMR × α → MA (mean ± std) + ASR (mean ± std) + AggTime (mean)
- 门槛达标/未达标标记
- 失败实验记录（如有）

**验证点**：
- 汇总表覆盖所有已完成实验
- 统计口径统一：mean ± std over seeds
- 门槛判定结果明确

---

### Step 9：编写项目 README.md

**文件**：`python/examples/federate/prebuilt_jobs/shieldfl/README.md`

**内容应包含**：

1. **项目概述**：一段话说明 ShieldFL/VeriFL-v16 在 FedML 上的攻防实验框架
2. **目录结构**：列出每个子目录和关键文件的职责
3. **快速上手**：
   - 环境要求（Python, FedML, MPI, CUDA）
   - 安装步骤
   - 运行第一个实验的命令
4. **实验矩阵**：支持的 model / dataset / attack / defense / aggregator 组合
5. **配置方式**：run_experiment.sh 参数说明（引用 Step 5 补充的用法文档）
6. **结果输出**：JSONL 格式说明，如何用 summarize_results.py 汇总
7. **关键设计决策**：
   - 攻击使用 FedML 内置实现
   - VeriFL 和 FedML 内置防御双重隔离
   - 分层均衡采样策略
8. **已知限制**：
   - Bulyan 不可用
   - cross-silo hierarchical 不支持
   - 仅 MPI backend

---

### Step 10：产出最终运行记录

**文件**：`RUN_RECORD_FINAL.md`（或在 PHASE2_GPU_RESULT.md 末尾追加）

**内容应包含**：

1. 最终冻结参数表
2. 实验执行硬件环境（GPU 型号、CUDA 版本、系统内存等）
3. 各里程碑达标情况（M1/M2/M3/M4 逐项核对）
4. 阈值校准记录（如有）
5. 偏差与例外记录
6. 最终结论：一段话说明项目达到的状态

---

## 4. Phase 3 明确不做的事

以下内容明确不属于 Phase 3 范围：

- 新增攻击或防御算法
- 修改 FedML 核心库代码
- DDP / DataParallel 单 client 多卡训练
- cross-cloud 部署
- Launch / MLOps 全接通
- DP / secure aggregation 兼容性验证
- CIFAR-100 / 其他数据集支持
- ShieldFL 原有攻击类迁移
- wandb 集成（JSONL 已满足需求）
- `eval/trust.py` TPR/FPR 实现（仅对 krum 有意义，对 VeriFL GA 语义不适用）

---

## 5. Phase 3 验收标准

### 5.1 实验完整性

- [ ] M1 全量 24 次实验完成，汇总表产出
- [ ] M2 核心子集完成，攻击效果可观测且跨 seed 方向一致
- [ ] M3 核心子集完成，防御效果可观测
- [ ] VeriFL 聚合器在至少一种攻击 × 一种 α 下的防御效果可与内置防御横向对比
- [ ] MNIST 任务线至少覆盖 M1 + M2 byzantine + M3 krum
- [ ] M4 汇总表覆盖全部已完成实验

### 5.2 参数统一性

- [ ] local epochs 全局统一且已冻结
- [ ] 实验参数表 (`EXPERIMENT_PARAMS.md`) 已写入仓库
- [ ] 编排脚本中无硬编码超参数与参数表矛盾

### 5.3 框架可用性

- [ ] run_experiment.sh 参数文档完善
- [ ] VeriFL + 内置防御的非法组合被拒绝
- [ ] 新实验只需修改脚本参数，不需要改代码
- [ ] summarize_results.py 可自动汇总全部 JSONL 结果

### 5.4 代码整洁性

- [ ] 历史配置文件移入 `config/legacy/`
- [ ] `_aggregate_bulyan()` 死代码已移除
- [ ] 历史结果文件移入 `results/legacy/`
- [ ] CPU 编排脚本标记为 legacy
- [ ] README.md 存在且内容准确

### 5.5 可选验收（推荐但不阻塞）

- [ ] CPU vs GPU 数值对齐验证完成
- [ ] Aggregation Time CPU vs GPU 对比完成
- [ ] `run_m2_final_gpu.sh` 和 `run_m3_final_gpu.sh` 就绪

---

## 6. 实施优先级

### P0（阻塞学术验收，必做）

1. **Step 1**：统一 epochs 并冻结参数
2. **Step 7**：补跑实验至核心子集完成
3. **Step 8**：运行 summarize_results.py 产出 M4 汇总表

### P1（框架质量，强烈建议）

4. **Step 3**：清理冗余配置文件
5. **Step 4**：清理冗余代码和结果
6. **Step 5**：完善 run_experiment.sh 文档和健壮性

### P2（完整性，推荐）

7. **Step 2**：补齐 MNIST 任务线
8. **Step 9**：编写 README.md
9. **Step 10**：产出最终运行记录

### P3（可选增强，时间允许时做）

10. **Step 6**：CPU/GPU 对齐与 Aggregation Time 基准

---

## 7. Phase 3 最终产出

Phase 3 结束时，仓库应新增或更新以下文件：

### 新增

- `EXPERIMENT_PARAMS.md`：冻结参数表
- README.md：项目使用说明
- `RESULTS_SUMMARY.md`：M4 汇总表
- `RUN_RECORD_FINAL.md`：最终运行记录
- `config/legacy/`：历史配置文件归档目录
- `results/legacy/`：历史结果文件归档目录
- `scripts/run_m2_final_gpu.sh`：M2 核心子集编排脚本
- `scripts/run_m3_final_gpu.sh`：M3 核心子集编排脚本

### 修改

- `scripts/run_experiment.sh`：补充文档 + 参数校验
- `trainer/baseline_aggregator.py`：移除 Bulyan 死代码
- `scripts/run_m1_baseline_gpu.sh`：epochs 统一
- `scripts/run_m2_priority_gpu.sh`：epochs 统一
- `scripts/run_m3_priority_gpu.sh`：epochs 统一

### 状态清理

- `config/fedml_config_*.yaml`（10 个历史配置）→ `config/legacy/`
- `results/` 下非正式实验 JSONL → `results/legacy/`
- CPU 编排脚本头部添加 legacy 标记

---

## 8. 一句话版 Phase 3

> **不再写新算法，不再加新功能——统一参数、补齐实验、清理遗留、固化结果，让仓库从"能跑"变成"好用"。**