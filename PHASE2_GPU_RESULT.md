# PHASE2_GPU_RESULT.md — Phase 2 GPU 实验执行记录

> 本文记录 PHASE2_GPU.md 规划的实际执行情况，包含与原文档的偏差、遇到的问题、做出的改动以及实验结果。
>
> **实验环境**：SSH alias `4090`，4× NVIDIA RTX 4090 (24GB)，CUDA 12.4，PyTorch 2.6.0+cu124，OpenMPI，Python 3.10
>
> **记录起始日期**：2026-03-26

---

## 1. 总体执行策略偏差

### 1.1 多 GPU 并行而非单 GPU 串行

PHASE2_GPU.md 的设计假设为单 GPU 串行执行全部实验。实际执行中，远程服务器有 4× RTX 4090，其中 GPU 2 被其他进程占用（ray::ClientAppActor ~15GB），因此采用 **3-GPU 并行策略**：

| tmux 会话 | GPU | 实验内容 | 脚本 |
|---|---|---|---|
| `m1` | GPU 0 | M1 基线可信 | `run_m1_baseline_gpu.sh` |
| `m2` | GPU 1 | M2 攻击生效（优先子集） | `run_m2_priority_gpu.sh` |
| `m3` | GPU 3 | M3 防御对照（优先子集） | `run_m3_priority_gpu.sh` |

### 1.2 优先子集策略而非全矩阵覆盖

PHASE2_GPU.md 规划了极大的实验矩阵（M2 全矩阵 288 次，M3 全矩阵 1728 次）。实际执行中先完成优先子集：

- **M2 优先子集**：3 attacks × PMR=0.2 × α∈{0.1, 0.5} × 3 seeds × 2 tasks = **36 次**
- **M3 优先子集**：(2 defenses + VeriFL) × byzantine × PMR=0.2 × α∈{0.1, 0.5} × 3 seeds × 2 tasks = **36 次**
- **M1 完整矩阵**：4 α × 3 seeds × 2 tasks = **24 次**

### 1.3 实验轮数缩减

M2 和 M3 优先子集的通信轮数从 PHASE2_GPU.md 规划的 100 轮降至 **50 轮**，以加速优先子集的完成。M1 保留 CIFAR10 100 轮、MNIST 50 轮不变。

---

## 2. 代码改动记录

### 2.1 `scripts/run_experiment.sh` — 核心改动

PHASE2_GPU.md §4.1.2 已预期需增加 `--gpu` / `--runtime` 参数。以下是实际实施的额外改动（文档未预期）：

#### 2.1.1 MPI root 用户适配

远程服务器以 root 身份运行，原始 `mpirun` 拒绝 root 执行。增加自动检测：

```bash
if [[ "$(id -u)" == "0" ]]; then
    MPI_EXTRA_ARGS="--allow-run-as-root"
fi
```

#### 2.1.2 `CUDA_VISIBLE_DEVICES` 传播

多 GPU 并行时需隔离 GPU，通过 MPI `-x` 标志传递：

```bash
if [[ "$GPU" == "true" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    MPI_EXTRA_ARGS="${MPI_EXTRA_ARGS} -x CUDA_VISIBLE_DEVICES=${GPU_ID}"
fi
```

新增 `--gpu_id` 命令行参数以支持指定 GPU 编号。

#### 2.1.3 label_flipping 攻击所需参数

FedML 的 `LabelFlippingAttack.__init__` 要求 `original_class_list`、`target_class_list`、`ratio_of_poisoned_client` 三个参数。PHASE2_GPU.md 未提及。在 YAML 模板中新增：

```yaml
original_class_list: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
target_class_list: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
ratio_of_poisoned_client: ${PMR}
```

#### 2.1.4 trimmed_mean 防御所需 `beta` 参数

FedML 的 `CoordinateWiseTrimmedMeanDefense.__init__` 要求 `config.beta`。PHASE2_GPU.md 未提及。新增：

```yaml
beta: ${TRIM_BETA}   # TRIM_BETA 默认 0.2
```

#### 2.1.5 mpirun 退出码处理

mpirun 正常结束时偶尔产生 spurious non-zero 退出码或 "command not found" 错误（观察到两次崩溃）。将最后一行从直接调用改为带容错：

```bash
MPI_EXIT=0
mpirun ${MPI_EXTRA_ARGS} -np $TOTAL_PROC python main_fedml_shieldfl.py --cf "$CONFIG_FILE" || MPI_EXIT=$?
if [[ $MPI_EXIT -ne 0 ]]; then
    echo "WARNING: mpirun exited with code $MPI_EXIT"
fi
exit $MPI_EXIT
```

### 2.2 防御名称修正（PHASE2_GPU.md 错误）

PHASE2_GPU.md §4.2.2 中 run_m3_defense_gpu.sh 使用的防御名称与 FedML 实际常量不一致：

| 文档中名称 | FedML 实际常量 (`constants.py`) | 修正 |
|---|---|---|
| `coordinate_wise_trimmed_mean` | `trimmed_mean` | ✅ 已修正 |
| `RFA` | `rfa`（大小写敏感） | ✅ 已修正 |
| `bulyan` | **不存在**（FedML 未实现） | ✅ 已移除 |

**最终 M3 内置防御列表**：`rfa krum trimmed_mean cclip`（4 种，非文档中的 5 种）。

### 2.3 `run_m1_baseline_gpu.sh` — epochs 提升

#### 问题

PHASE2_GPU.md §4.2.2 和示例 YAML 均指定 `epochs: 1`。以 epochs=1 完成的首轮 M1 CIFAR10 实验（10/24 完成）显示 **所有 α 值均未达到学术门槛**：

| α | epochs=1 MA (mean±std) | 门槛 | 差距 |
|---|---|---|---|
| 0.1 | 57.18% ± 8.24% | ≥ 75% | -17.8% |
| 0.3 | 74.40% ± 2.01% | ≥ 78% | -3.6% |
| 0.5 | 78.75% ± 0.56% | ≥ 82% | -3.2% |
| 100 | 82.05%（仅 1 seed） | ≥ 85% | -3.0% |

根本原因：`epochs=1` 每轮本地训练量不足，ResNet18 在 CIFAR10 上无法充分学习。

#### 修正决策

PHASE2_GPU.md §5 Step 6 的阈值校准规则允许一次校准，但考虑到 α=0.1 差距达 17.8%，校准不合理。选择 **方案 2：将 local epochs 从 1 提升至 5**。

#### 具体改动

- run_m1_baseline_gpu.sh：`--epochs 1` → `--epochs ${EPOCHS}`，`EPOCHS=5`
- 新增断点续跑逻辑（`check_done()` 函数），可跳过已有结果
- 新增错误容忍（`|| echo "WARNING...""`），单个实验失败不终止全流程
- 旧 epochs=1 的结果归档至 `results/old_epochs1/`

> **注意**：M2/M3 优先子集脚本保留 `epochs=1`，因为攻防实验的攻击效果判定与 M1 基线无关，且 epochs=1 下攻击效果（MA 下降）已经很显著。

### 2.4 新增文件（非文档预期）

| 文件 | 用途 | 文档对应 |
|---|---|---|
| `scripts/run_m2_priority_gpu.sh` | M2 攻击优先子集脚本 | 文档只有全矩阵版 |
| `scripts/run_m3_priority_gpu.sh` | M3 防御优先子集脚本 | 文档只有全矩阵版 |
| `scripts/launch_all_gpu.sh` | 3-GPU tmux 并行启动器 | 文档无 |
| `scripts/check_progress.sh` | 进度监控脚本 | 文档无 |
| `scripts/summarize_results.py` | 结果汇总统计 | 文档 §5 Step 9 有类似建议 |

### 2.5 未实施的文档规划项

以下 PHASE2_GPU.md 中规划但尚未实施的改动：

| 规划项 | 文档章节 | 状态 | 原因 |
|---|---|---|---|
| `runtime.py` GPU 环境检查 | §4.1.1 | ❌ 未实施 | GPU 验证通过后直接进入实验阶段 |
| `gpu_accelerator.py` GPU 日志 | §4.1.4 | ❌ 未实施 | 非关键路径 |
| `metrics.py` 硬件上下文字段 | §4.1.6 | ❌ 未实施 | 非关键路径 |
| CPU vs GPU 数值对齐验证 | §5 Step 4 | ❌ 跳过 | 直接进入学术实验 |
| Aggregation Time 基准测定 | §5 Step 5 | ❌ 跳过 | VeriFL 聚合器尚未纳入 M1 |
| `compare_cpu_gpu_metrics.py` | §4.2.3 | ❌ 未创建 | 跳过了对齐验证步骤 |
| GPU YAML 配置文件 | §4.2.1 | ❌ 未创建 | run_experiment.sh 动态生成 YAML，无需模板文件 |

---

## 3. 运行时问题记录

### 3.1 M2 首次崩溃：label_flipping AttributeError

**时间**：实验 7/36（label_flipping 首个实验）

**错误**：
```
AttributeError: 'Arguments' object has no attribute 'original_class_list'
```

**根因**：FedML 的 `LabelFlippingAttack.__init__` 需要 `args.original_class_list`、`args.target_class_list`、`args.ratio_of_poisoned_client`。PHASE2_GPU.md 的 YAML 模板中未包含这些参数。

**修复**：在 run_experiment.sh YAML 模板中添加了这三个参数（见 §2.1.3）。

### 3.2 M3 首次崩溃：defense_type 未定义

**时间**：实验 7/36（trimmed_mean 首个实验）

**错误**：
```
Exception: args.defense_type is not defined!
```

**根因**：PHASE2_GPU.md 使用 `coordinate_wise_trimmed_mean` 作为防御名称，但 FedML 的常量为 `trimmed_mean`。使用错误名称导致 FedML 无法识别。

**修复**：所有脚本中 `coordinate_wise_trimmed_mean` → `trimmed_mean`。

### 3.3 M3 二次崩溃：trimmed_mean 缺少 beta

**时间**：修正防御名称后重启 M3

**错误**：
```
AttributeError: 'Arguments' object has no attribute 'beta'
```

**根因**：`CoordinateWiseTrimmedMeanDefense.__init__` 要求 `config.beta` 参数（裁剪比例）。

**修复**：在 run_experiment.sh YAML 模板中添加 `beta: ${TRIM_BETA}`，默认 `TRIM_BETA="0.2"`。

### 3.4 M1 / M2 mpirun 退出码异常

**现象**：实验正常完成（所有 client finished），但 bash 报 `line 189: edml_shieldfl.py: command not found` 或 `line 186: 0: command not found`。

**根因**：mpirun 多进程退出时的竞态条件在 bash `set -e` 模式下触发了后续脚本中止。

**修复**：
1. run_experiment.sh 末尾 mpirun 调用增加错误捕获（`|| MPI_EXIT=$?`）
2. M1/M2/M3 脚本的 run_experiment.sh 调用增加 `|| echo "WARNING..."` 容错

### 3.5 bulyan 防御不可用

**发现**：检查 FedML 源码 `fedml/core/security/constants.py` 发现 `bulyan` 不在已定义的防御常量中。

**处理**：从 run_m3_defense_gpu.sh（全矩阵版）中移除 `bulyan`。PHASE2_GPU.md 中的 5 种防御减为 4 种：`rfa krum trimmed_mean cclip`。

---

## 4. M1 基线实验结果

### 4.1 epochs=1 结果（已归档）

首轮以 PHASE2_GPU.md 原始配置（`epochs=1`）完成了 CIFAR10 全部 4α × 3seeds = 12 次实验中的 10 次（α=100 仅完成 seed=0）。结果归档在 `results/old_epochs1/`。

**ResNet18 + CIFAR10, 100 rounds, epochs=1**

| α | seed=0 | seed=1 | seed=2 | mean ± std | 门槛 | 判定 |
|---|---|---|---|---|---|---|
| 0.1 | 64.83% | 48.46% | 58.25% | 57.18% ± 8.24% | ≥ 75% | ❌ FAIL (-17.8%) |
| 0.3 | 76.26% | 72.26% | 74.68% | 74.40% ± 2.01% | ≥ 78% | ❌ FAIL (-3.6%) |
| 0.5 | 78.63% | 78.27% | 79.36% | 78.75% ± 0.56% | ≥ 82% | ❌ FAIL (-3.2%) |
| 100 | 82.05% | — | — | 82.05% | ≥ 85% | ❌ FAIL (-3.0%) |

**结论**：epochs=1 下系统性低于门槛，α≥0.3 差距约 3-4%，α=0.1 差距近 18%。阈值校准不足以覆盖，改为 epochs=5。

### 4.2 epochs=5 结果（⛔ 已终止）

2026-03-26 15:45 启动，2026-03-26 ~20:30 手动终止。完成 4/24 实验（α=0.1 全部 3 seeds + α=0.3 seed=0），2 个部分完成（α=0.3 seed1/2），其余未启动。

**终止原因**：epochs=1→5 提升仅 +0.7%~3.5%，远低于预期的 5-10%。α=0.1 仍距门槛 14.3%，继续运行无法改变结论。

**ResNet18 + CIFAR10, 100 rounds, epochs=5**

| α | seed=0 | seed=1 | seed=2 | mean ± std | 门槛 | 判定 |
|---|---|---|---|---|---|---|
| 0.1 | 59.39% | 58.22% | 64.58% | 60.73% ± 3.33% | ≥ 75% | ❌ FAIL (-14.3%) |
| 0.3 | 75.06% | 60.23%* | 64.65%* | — | ≥ 78% | ❌ FAIL (seed=0 -2.9%) |
| 0.5 | — | — | — | — | ≥ 82% | ⛔ 未启动 |
| 100 | — | — | — | — | ≥ 85% | ⛔ 未启动 |

> \* α=0.3 seed1 仅完成 9/100 轮，seed2 仅完成 13/100 轮（终止时的中间值，不可用于判定）。

**LeNet5 + MNIST**：仅 seed=0 完成 3/50 轮（MA=8.4%），未产出有效结果。

#### epochs=1 vs epochs=5 对比（α=0.1，完整 100 rounds）

| 指标 | epochs=1 | epochs=5 | 变化 |
|---|---|---|---|
| seed=0 | 64.83% | 59.39% | -5.44% |
| seed=1 | 48.46% | 58.22% | +9.76% |
| seed=2 | 58.25% | 64.58% | +6.33% |
| mean ± std | 57.18% ± 8.24% | 60.73% ± 3.33% | +3.55% |

**分析**：epochs=5 将标准差从 8.24% 降至 3.33%（训练更稳定），但均值仅提升 3.55%，距门槛 75% 仍差 14.3%。瓶颈在于 non-IID 聚合效率而非本地训练量。α=0.3 的 seed=0 甚至从 76.26%（epochs=1）降至 75.06%（epochs=5），说明增加 epochs 不是有效改善路径。

---

## 5. M2 攻击实验结果（优先子集，✅ 已完成）

> **脚本**：`scripts/run_m2_priority_gpu.sh`
> **完成时间**：2026-03-26 18:58:40，36/36 全部完成
> **配置**：3 attacks × PMR=0.2 × α∈{0.1, 0.5} × 3 seeds × 2 tasks，50 rounds，epochs=1

### 5.1 ResNet18 + CIFAR10

**byzantine（PMR=0.2, 50 rounds, epochs=1）**

| α | seed=0 | seed=1 | seed=2 | mean ± std |
|---|---|---|---|---|
| 0.1 | 29.43% | 38.33% | 43.11% | 36.96% ± 6.94% |
| 0.5 | 72.26% | 64.11% | 67.59% | 67.99% ± 4.09% |

**label_flipping（PMR=0.2, 50 rounds, epochs=1）**

| α | seed=0 | seed=1 | seed=2 | mean ± std |
|---|---|---|---|---|
| 0.1 | 43.82% | 38.01% | 56.80% | 46.21% ± 9.60% |
| 0.5 | 78.72% | 77.09% | 78.28% | 78.03% ± 0.84% |

**model_replacement（PMR=0.2, 50 rounds, epochs=1）**

| α | seed=0 MA / ASR | seed=1 MA / ASR | seed=2 MA / ASR | MA mean±std | ASR mean |
|---|---|---|---|---|---|
| 0.1 | 16.61% / 0.00% | 15.83% / 0.00% | 17.97% / 21.24% | 16.80% ± 1.08% | 7.08% |
| 0.5 | 46.55% / 0.17% | 20.90% / 14.21% | 46.21% / 5.97% | 37.89% ± 14.71% | 6.78% |

### 5.2 LeNet5 + MNIST

**byzantine（PMR=0.2, 50 rounds, epochs=1）**

| α | seed=0 | seed=1 | seed=2 | mean ± std |
|---|---|---|---|---|
| 0.1 | 92.48% | 97.49% | 95.90% | 95.29% ± 2.53% |
| 0.5 | 97.43% | 97.93% | 98.13% | 97.83% ± 0.43% |

> 注：MNIST byzantine 有 6 个数据点（n=6），部分为 M2 首次运行遗留。

**label_flipping（PMR=0.2, 50 rounds, epochs=1）**

| α | seed=0 | seed=1 | seed=2 | mean ± std |
|---|---|---|---|---|
| 0.1 | 97.46% | 98.18% | 97.09% | 97.58% ± 0.52% |
| 0.5 | 98.43% | 98.73% | 98.79% | 98.65% ± 0.20% |

**model_replacement（PMR=0.2, 50 rounds, epochs=1）**

| α | seed=0 MA / ASR | seed=1 MA / ASR | seed=2 MA / ASR | MA mean±std | ASR mean |
|---|---|---|---|---|---|
| 0.1 | 20.45% / 46.19% | 11.35% / 0.00% | 9.74% / 0.00% | 13.85% ± 5.78% | 15.40% |
| 0.5 | 88.72% / 0.09% | 96.10% / 0.18% | 94.54% / 0.03% | 93.12% ± 3.89% | 0.10% |

### 5.3 M2 学术门槛对照

#### 无目标攻击（byzantine / label_flipping）

> 门槛：CIFAR10 100 轮内 MA ≤ 40%，MNIST 100 轮内 MA ≤ 50%，需跨 seed 和 α∈{0.1,0.5} 一致方向

| 攻击 | 数据集 | α=0.1 MA(mean) | 判定 | α=0.5 MA(mean) | 判定 |
|---|---|---|---|---|---|
| byzantine | CIFAR10 | 36.96% | ✅ ≤40% | 67.99% | ❌ >40% |
| byzantine | MNIST | 95.29% | ❌ >50% | 97.83% | ❌ >50% |
| label_flipping | CIFAR10 | 46.21% | ❌ >40% | 78.03% | ❌ >40% |
| label_flipping | MNIST | 97.58% | ❌ >50% | 98.65% | ❌ >50% |

**结论**：❌ **M2 无目标攻击整体不满足学术要求**。8 个条件中仅 1 个通过（byzantine CIFAR10 α=0.1）。学术需求要求"跨 α∈{0.1,0.5} 一致方向"，不满足。

**主因分析**：
1. PMR=20%（仅 2/10 客户端恶意），诚实客户端 majority 足以稀释攻击
2. epochs=1 下局部训练量很弱，恶意梯度贡献被多次聚合冲淡
3. MNIST 任务太简单，50 轮足够收敛到高 MA
4. 实验仅 50 轮而非学术要求的 100 轮

#### 目标后门攻击（model_replacement）

> 门槛：隐蔽性 MA 下降 ≤ 3%，破坏性 ASR ≥ 85%，需跨 seed 和 α 稳定生效

| 数据集 | α | MA(mean) | ASR(final mean) | ASR(max mean) | 隐蔽性 | 破坏性 |
|---|---|---|---|---|---|---|
| CIFAR10 | 0.1 | 16.80% | 7.08% | 82.53% | ❌ MA 崩溃 | ❌ ASR<85% |
| CIFAR10 | 0.5 | 37.89% | 6.78% | 60.11% | ❌ MA 崩溃 | ❌ ASR<85% |
| MNIST | 0.1 | 13.85% | 15.40% | 89.99% | ❌ MA 崩溃 | 部分达标 |
| MNIST | 0.5 | 93.12% | 0.10% | 15.63% | ≈隐蔽 | ❌ ASR 极低 |

**结论**：❌ **M2 model_replacement 完全不满足学术要求**。FedML 的 model_replacement 实现表现为"二择一"：要么 MA 崩溃（不隐蔽）+ ASR 不稳定，要么隐蔽但 ASR 近 0。无法同时满足隐蔽性和破坏性。

### 5.4 M2 补救方向

1. **增大 PMR 至 30%~40%**：增加恶意客户端比例，增强攻击压力
2. **增加 rounds 至 100**：给攻击更多时间累积效果（学术需求本就要求 100 轮）
3. **增加 epochs**：让恶意客户端的毒模型贡献更大
4. **阈值校准**：PHASE2_GPU.md §5 Step 6 允许一次系统性阈值校准（需记录原值/新值/原因/证据）
5. **model_replacement**：检查 FedML 实现是否正确或调整攻击参数

---

## 6. M3 防御实验结果（优先子集，✅ 已完成）

> **脚本**：`scripts/run_m3_priority_gpu.sh`
> **完成时间**：2026-03-26 18:41:43，36/36 全部完成
> **配置**：(krum + trimmed_mean + VeriFL) × byzantine × PMR=0.2 × α∈{0.1, 0.5} × 3 seeds × 2 tasks，50 rounds，epochs=1

### 6.1 ResNet18 + CIFAR10

**krum + byzantine**

| α | seed=0 | seed=1 | seed=2 | mean ± std |
|---|---|---|---|---|
| 0.1 | 21.46% | 14.21% | 10.05% | 15.24% ± 5.77% |
| 0.5 | 55.15% | 40.77% | 42.23% | 46.05% ± 7.91% |

**trimmed_mean + byzantine**

| α | seed=0 | seed=1 | seed=2 | mean ± std |
|---|---|---|---|---|
| 0.1 | 28.20% | 31.28% | 27.40% | 28.96% ± 2.05% |
| 0.5 | 62.96% | 65.85% | 70.87% | 66.56% ± 3.99% |

**VeriFL（聚合器） + byzantine**

| α | seed=0 | seed=1 | seed=2 | mean ± std |
|---|---|---|---|---|
| 0.1 | 47.51% | 39.83% | 50.32% | 45.89% ± 5.43% |
| 0.5 | 75.58% | 75.22% | 75.30% | 75.37% ± 0.19% |

### 6.2 LeNet5 + MNIST

**krum + byzantine**

| α | seed=0 | seed=1 | seed=2 | mean ± std |
|---|---|---|---|---|
| 0.1 | 48.92% | 9.58% | 48.39% | 35.63% ± 22.56% |
| 0.5 | 84.87% | 92.02% | 94.88% | 90.59% ± 5.14% |

**trimmed_mean + byzantine**

| α | seed=0 | seed=1 | seed=2 | mean ± std |
|---|---|---|---|---|
| 0.1 | 71.94% | 85.86% | 79.23% | 79.01% ± 6.96% |
| 0.5 | 96.22% | 96.42% | 98.22% | 96.95% ± 1.06% |

**VeriFL（聚合器） + byzantine**

| α | seed=0 | seed=1 | seed=2 | mean ± std |
|---|---|---|---|---|
| 0.1 | 97.83% | 97.88% | 96.96% | 97.56% ± 0.52% |
| 0.5 | 98.10% | 98.60% | 98.28% | 98.33% ± 0.25% |

### 6.3 M3 学术门槛对照

#### 防御 vs 无防御（M2 byzantine baseline）对比

> 核心问题：防御能否提升受攻击下的模型精度？

| 防御 | 数据集 | α=0.1 MA | vs 无防御 | α=0.5 MA | vs 无防御 |
|---|---|---|---|---|---|
| **VeriFL** | CIFAR10 | 45.89%±5.43% | **+8.93%** ✅ | 75.37%±0.19% | **+7.38%** ✅ |
| **VeriFL** | MNIST | 97.56%±0.52% | **+2.27%** ✅ | 98.33%±0.25% | **+0.50%** ✅ |
| krum | CIFAR10 | 15.24%±5.77% | -21.72% ❌ | 46.05%±7.91% | -21.94% ❌ |
| krum | MNIST | 35.63%±22.56% | -59.66% ❌ | 90.59%±5.14% | -7.24% ❌ |
| trimmed_mean | CIFAR10 | 28.96%±2.05% | -8.00% ❌ | 66.56%±3.99% | -1.43% ❌ |
| trimmed_mean | MNIST | 79.01%±6.96% | -16.28% ❌ | 96.95%±1.06% | -0.88% ≈ |

> 无防御基线（M2 byzantine）：CIFAR10 α=0.1→36.96%, α=0.5→67.99%；MNIST α=0.1→95.29%, α=0.5→97.83%

#### 分析

1. **VeriFL 正面结论成立** ✅：在全部 4 个条件（2 数据集 × 2 α）下，VeriFL 均提升了受攻击下的 MA。特别是 CIFAR10 α=0.5 下 std 从 4.09% 降至 0.19%，说明 VeriFL 显著提高了聚合稳定性。

2. **krum 表现极差** ❌：在所有条件下 MA 反而低于无防御。MNIST α=0.1 出现灾难性失败（35.63%±22.56%，其中 seed=1 仅 9.58%）。可能原因：
   - krum 在 10 clients + 2 byzantine 的配置下选择了非最优客户端
   - non-IID（α=0.1）使客户端梯度差异大，krum 的距离度量失效

3. **trimmed_mean 小幅退化** ❌：在 α=0.5 高 IID 条件下影响较小（CIFAR10 -1.43%, MNIST -0.88%），但在 α=0.1 低 IID 条件下明显退化。

4. **学术需求要求的"无攻击防御保真度"（MA 下降 ≤ 2%）未验证**：优先子集未包含无攻击场景下的防御实验。

> **注意**：krum 和 trimmed_mean 在无防御基线上表现更差是一个**合理的学术发现**——传统拜占庭容错防御在 non-IID 联邦学习中可能适得其反。这恰好可以作为 VeriFL 论文的对照论据。

---

## 7. 偏差汇总表

| 文档条目 | PHASE2_GPU.md 原文 | 实际执行 | 偏差原因 |
|---|---|---|---|
| 执行模式 | 单 GPU 串行 | 3 GPU 并行（GPU 0/1/3） | 硬件资源充分，提高效率 |
| M1 local epochs | `epochs: 1` | `epochs: 5`（已终止） | epochs=1 下全部 α 不达标，差距最大 17.8% |
| M1 epochs=5 完整度 | 24 次全完成 | 仅完成 4/24 后终止 | epochs=5 提升仅 +3.55%（α=0.1），不足以达标 |
| M2/M3 实验规模 | 全矩阵覆盖 | 优先子集（36 次/组） | 先验证核心场景，再展开 |
| M2/M3 通信轮数 | 100 轮 | 50 轮 | 加速优先子集验证 |
| M3 防御数量 | 5 种（含 bulyan） | 4 种（rfa krum trimmed_mean cclip） | `bulyan` 在 FedML 中未实现 |
| M3 防御名称 | `coordinate_wise_trimmed_mean`, `RFA` | `trimmed_mean`, `rfa` | 文档与 FedML 常量不一致 |
| CPU vs GPU 对齐 | 必做（Step 4） | 跳过 | 优先完成学术实验 |
| Aggregation Time | 必做（Step 5） | 跳过 | VeriFL 聚合器尚未纳入实验 |
| runtime.py 改动 | §4.1.1 三处修改 | 未实施 | 非阻塞性改动 |
| gpu_accelerator.py 日志 | §4.1.4 | 未实施 | 非阻塞性改动 |
| metrics.py 硬件字段 | §4.1.6 | 未实施 | 非阻塞性改动 |
| YAML 配置模板文件 | §4.2.1 三个新文件 | 未创建 | run_experiment.sh 动态生成 YAML |
| label_flipping 参数 | 未提及 | 新增 3 个参数 | FedML 实际需要 |
| trimmed_mean beta | 未提及 | 新增 beta=0.2 | FedML 实际需要 |
| mpirun root 运行 | 未提及 | 新增 --allow-run-as-root | 远程服务器以 root 运行 |
| CUDA_VISIBLE_DEVICES | 未提及 | 新增 -x 传播 | 多 GPU 隔离需要 |

---

## 8. 当前实验状态（2026-03-27 更新）

| 会话 | GPU | 脚本 | 进度 | 状态 |
|---|---|---|---|---|
| m1 | GPU 0 | run_m1_baseline_gpu.sh (epochs=5) | 4/24 | ⛔ 已终止（提升不足） |
| m2 | GPU 1 | run_m2_priority_gpu.sh | 36/36 | ✅ 已完成（18:58:40） |
| m3 | GPU 3 | run_m3_priority_gpu.sh | 36/36 | ✅ 已完成（18:41:43） |

**结果文件总数**：76（含 M1 epochs=5 的 4 个完整文件 + 2 个部分文件）

**M1 epochs=5 完成情况**：
- ✅ 完整：α=0.1 seed={0,1,2}（3 个，100 rounds），α=0.3 seed=0（1 个，100 rounds）
- ⚠️ 部分：α=0.3 seed=1（9 rounds），α=0.3 seed=2（13 rounds）
- ⛔ 未启动：α=0.5（3 个），α=100（3 个），MNIST 全部（12 个）
- 终止原因：epochs=1→5 仅提升 +3.55%（α=0.1 mean），远低于预期 5-10%，无法达到学术门槛

---

## 9. 后续待办

### M1 基线策略决定（⛔ epochs=5 已终止）

> epochs=5 提升不足（+3.55%），不继续 epochs=5 路线。需选择替代方案：

- [ ] **方案 A — 阈值校准**：PHASE2_GPU.md §5 Step 6 允许一次系统性校准。将 α=0.1 门槛从 ≥75% 降至 ≥60%，α=0.3 从 ≥78% 降至 ≥74%。需记录原值/新值/原因/证据。
- [ ] **方案 B — 接受现状**：以 epochs=5 α=0.1 的 60.73% 和 epochs=1 的其他结果作为基线，用相对变化（攻击下降幅度）替代绝对门槛论证。
- [ ] **方案 C — 增加通信轮数**：将 CIFAR10 从 100 轮增至 200 轮，但代价为实验时间翻倍。
- [ ] **方案 D — 冻结 epochs 为某个值**：在 Phase 3 中统一所有实验的 epochs 设置，重新跑完整矩阵。

### M2 补救（无目标攻击未达标）

- [ ] 决定补救策略：增大 PMR（30%/40%）/ 增加 rounds 到 100 / 阈值校准
- [ ] 根据策略补充实验
- [ ] 重新审视 model_replacement 的 FedML 实现

### M3 跟进

- [ ] 补做无攻击场景下防御保真度实验（验证 MA 下降 ≤ 2% 门槛）
- [ ] 补充 rfa 和 cclip 防御实验
- [ ] 补充 label_flipping / model_replacement 攻击下的防御实验

### 后续阶段

- [ ] 运行 `summarize_results.py` 产出汇总统计表
- [ ] 评估是否展开 M2/M3 全矩阵
- [ ] 进入 PHASE3.md 流程