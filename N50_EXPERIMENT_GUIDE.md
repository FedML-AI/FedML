# N=50 正式实验全流程文档

> **创建日期**：2026-04-09  
> **硬件环境**：ssh 4090 | 4×RTX 4090 (24564 MiB/卡) | GPU 0 被占用  
> **实验基础**：M2_SA_FIX_PHASE1_ERROR2_FIX.md 冻结的 28 组实验矩阵  
> **代码基线**：git commit `420ff28d`

---

## 一、环境和修复概览

### 1.1 本次 session 修复的 bug（共 5 个）

| # | Bug | 文件 | 修复 |
|:--|:----|:-----|:----|
| 1 | FedML 聚合器将全部 50 个客户端模型移至 GPU 导致 OOM | `fedml/cross_silo/server/fedml_aggregator.py` | `model_params_to_device(... torch.device("cpu"))` |
| 2 | FedML `__init__.py` 无条件覆盖 `cpu_transfer=False` (MPI 后端) | `fedml/__init__.py:296` | `getattr(args, "cpu_transfer", False)` 保留 YAML 设置 |
| 3 | 共享 CUDA_VISIBLE_DEVICES 导致跨 GPU CUDA context 污染 (~428 MiB/进程) | `scripts/gpu_wrapper.sh` (新建) | 每个 MPI 进程仅可见分配给它的单张 GPU |
| 4 | FedML 内置 `FedMLDefender` 不识别 "shieldfl" 自定义防御 | `scripts/run_experiment.sh` | `enable_defense: false` for shieldfl |
| 5 | PyTorch `__setitem__` 混合 advanced+basic indexing 维度重排导致 trigger 注入失败 | `trainer/verifl_trainer.py:193-198` | 改用逐样本循环 `for _pi in indices` |

### 1.2 GPU 分配方案

| 配置 | GPU 使用 | 进程映射 | 峰值显存 |
|:-----|:--------|:---------|:---------|
| CIFAR-10 (ResNet18, N=50) | GPU 1,2,3 | 13+19+19 | GPU1≈15.9GB, GPU2≈20.7GB, GPU3≈20.4GB |
| MNIST (LeNet5, N=50) | GPU 1,2 | 26+25 | GPU1≈9.3GB, GPU2≈9.3GB |

### 1.3 Smoke Test 结果

| ID | 配置 | 结果 |
|:---|:-----|:-----|
| S0-1 | CIFAR-10 no-attack, 3 rounds | ✅ PASS (10%→10.1%→22.3%) |
| S0-2 | MNIST no-attack, 3 rounds | ✅ PASS (12.1%→28%→61.7%) |
| S0-4 | CIFAR-10 model_replacement + shieldfl, 3 rounds | ✅ PASS (ASR=100%, acc=10%) |
| S0-5 | Dirichlet α=0.1 分区验证 | ✅ PASS (ratio=74.4) |

---

## 二、实验矩阵（28 组）

### 2.1 主实验（24 组）

**CIFAR-10 + ResNet-18**（100 rounds, 3 GPUs, ~50 min/exp）：

| α | seed=0 | seed=1 | seed=2 |
|:--|:-------|:-------|:-------|
| 0.1 | c10_atk_a0.1_s0 | c10_atk_a0.1_s1 | c10_atk_a0.1_s2 |
| 0.5 | c10_atk_a0.5_s0 | c10_atk_a0.5_s1 | c10_atk_a0.5_s2 |
| 100 | c10_atk_a100_s0 | c10_atk_a100_s1 | c10_atk_a100_s2 |

**MNIST + LeNet-5**（50 rounds, 2 GPUs, ~20 min/exp）：

| α | seed=0 | seed=1 | seed=2 | seed=3 | seed=4 |
|:--|:-------|:-------|:-------|:-------|:-------|
| 0.1 | mn_atk_a0.1_s0 | mn_atk_a0.1_s1 | mn_atk_a0.1_s2 | mn_atk_a0.1_s3 | mn_atk_a0.1_s4 |
| 0.5 | mn_atk_a0.5_s0 | mn_atk_a0.5_s1 | mn_atk_a0.5_s2 | mn_atk_a0.5_s3 | mn_atk_a0.5_s4 |
| 100 | mn_atk_a100_s0 | mn_atk_a100_s1 | mn_atk_a100_s2 | mn_atk_a100_s3 | mn_atk_a100_s4 |

**共性参数**：attack=model_replacement, defense=none, PMR=20%, epochs=1, batch_size=64, lr=0.01, scale_gamma=auto, backdoor_per_batch=20, 每轮持续攻击

### 2.2 控制组（4 组）

| TAG | 数据集 | PMR | α | seed | γ | 目的 |
|:----|:------|:----|:--|:-----|:--|:-----|
| c10_ctrl_gamma1 | CIFAR-10 | 20% | 100 | 0 | 1 | γ=1 因果性控制 |
| mn_ctrl_gamma1 | MNIST | 20% | 100 | 0 | 1 | γ=1 因果性控制 |
| c10_baseline_noatk | CIFAR-10 | 0% | 0.5 | 0 | — | 无攻击 FedAvg 基线 |
| mn_baseline_noatk | MNIST | 0% | 0.5 | 0 | — | 无攻击 FedAvg 基线 |

---

## 三、执行指南

### 3.1 启动全部 28 组

```bash
ssh 4090
cd /data/home/ykdz/FedML/python/examples/federate/prebuilt_jobs/shieldfl
source /data/home/ykdz/FedML/.venv/bin/activate

# 在 tmux 中运行（防止断连）
tmux new-session -s batch
nohup bash scripts/batch_n50.sh > results/batch_runner.log 2>&1 &
# 或直接在 tmux 中：
bash scripts/batch_n50.sh 2>&1 | tee results/batch_runner.log
```

### 3.2 监控进度

```bash
# 查看已完成实验
cat results/batch_done.txt

# 查看当前实验日志
ls -lt results/batch_logs/*.log | head -3
tail -f results/batch_logs/<latest>.log

# GPU 使用率
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# 查看 JSONL 结果
ls -la results/metrics_*.jsonl
```

### 3.3 断点续跑

脚本支持断点续跑。已完成的实验记录在 `results/batch_done.txt`，重新启动脚本会自动跳过已完成的任务。

### 3.4 单独重跑某组

```bash
# 删除对应记录
sed -i '/c10_atk_a0.1_s0/d' results/batch_done.txt
# 重新启动脚本
bash scripts/batch_n50.sh
```

---

## 四、输出文件

### 4.1 JSONL 指标文件

每个实验产生一个 JSONL，路径：
```
results/metrics_{MODEL}_{DATASET}_shieldfl_atk{ATTACK}_def{DEFENSE}_a{ALPHA}_pmr{PMR}_seed{SEED}.jsonl
```

每行一个 round，包含字段：
- `round`, `test_accuracy`, `test_loss`, `test_total`
- `asr`, `max_asr`（攻击实验）
- `gamma_actual`, `malicious_count`, `trigger_value_normalized`
- `agg_time`, `timestamp`
- 环境元信息（model, dataset, alpha, device, cuda_version 等）

### 4.2 配置快照

每个实验的完整 YAML 配置保存在：
```
results/configs/config_{MODEL}_{DATASET}_shieldfl_atk{ATTACK}_def{DEFENSE}_a{ALPHA}_pmr{PMR}_seed{SEED}.yaml
```

### 4.3 批量日志

```
results/batch_logs/{TAG}.log     — 每个实验的完整 stdout/stderr
results/batch_done.txt            — 完成状态记录
results/batch_runner.log          — 批量脚本主日志
```

---

## 五、验收标准摘要（来源：M2_SA_FIX_PHASE1_ERROR2_FIX.md §6）

| AC | 条件 | 阈值 |
|:---|:-----|:-----|
| AC-A-1 | FedAvg 无防御下 ASR（median over seeds） | CIFAR-10 ≥ 80%, MNIST ≥ 90% |
| AC-A-2 | γ=1 控制组 ASR 显著低于 γ=auto | 差异 > 20pp |
| AC-A-3 | 无攻击基线 ASR ≈ 1/C | ≤ 15% |
| AC-B-1 | Clean accuracy 不完全崩塌 | 已攻击 acc > random (10% CIFAR, 10% MNIST) |

---

## 六、修改文件汇总

| 文件（相对 FedML/python 根） | 修改内容 |
|:---------------------------|:---------|
| `fedml/__init__.py` | 保留 cpu_transfer YAML 设置 |
| `fedml/cross_silo/server/fedml_aggregator.py` | CPU 端模型聚合 |
| `examples/.../shieldfl/scripts/gpu_wrapper.sh` | 新建：每进程 CUDA 隔离 |
| `examples/.../shieldfl/scripts/run_experiment.sh` | GPU wrapper 集成、defense 修复 |
| `examples/.../shieldfl/scripts/batch_n50.sh` | 新建：28 组批量执行 |
| `examples/.../shieldfl/config/gpu_mapping.yaml` | mapping_50clients_isolated |
| `examples/.../shieldfl/trainer/verifl_trainer.py` | 循环 trigger 注入 |
