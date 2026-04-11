# ShieldFL → FedML Phase 2 GPU 验证实施规划

> 本文基于以下上游文档推导：
> - **MOVE.md §5.1**：`single-gpu-deterministic` 与 `multi-gpu-throughput` 两档 GPU 运行模式定义
> - **PHASE1.md**：CPU 确定性约束框架与设备治理原则
> - **PHASE2.md**：已交付的 CPU 攻防全链路基础设施
> - **学术需求.md**：M1–M4 学术门槛、Aggregation Time 硬件公平性、`mean ± std` 统计口径
> - **RUN_RECORD_PHASE1.md**：Phase 1 CPU 验证记录
>
> Phase 2 GPU 的关键词：**在 Phase 2 CPU 已验证的攻防全链路上，将运行档位从 cpu-deterministic 升级到 single-gpu-deterministic 和 multi-gpu-throughput，完成 GPU 环境下的数值对齐验证、性能基准测定和学术规模实验。**

---

## 1. Phase 2 GPU 的目标

Phase 2 GPU 必须同时满足三件事：

1. **数值对齐目标**：在 `single-gpu-deterministic` 模式下，证明 GPU 运行与 CPU 运行在相同 seed / 相同 client 顺序 / 相同超参数下，精度指标（MA / Loss）的差异在可接受容差内（`|Δ MA| ≤ 0.5%`），且算法三阶段执行路径完全一致。
2. **性能基准目标**：在 GPU 环境下获得可量化的 Aggregation Time 度量，证明 `GPUAccelerator` 的 fitness 评估和 BN 校准路径真正利用了 GPU 加速，并与 CPU 基线形成可比对的计时数据。
3. **学术规模目标**：在 GPU 环境下完成学术需求.md 所要求的正式实验矩阵（attack × defense × PMR × α × seed），产出结构化逐轮指标，满足 M1–M3 的学术门槛。

Phase 2 GPU 结束时的状态应为：

> **所有 M1/M2/M3 实验可在 GPU 环境下以学术要求的规模和通信轮次完成，逐轮指标已落盘，Aggregation Time 可对比，VeriFL 与 FedAvg/内置防御在同硬件口径下横向可比。**

---

## 2. 前置条件

### 2.1 Phase 2 CPU 已交付能力

以下能力已在 CPU 环境下验证通过，Phase 2 GPU 可直接复用：

| 能力 | 文件 | CPU 验证状态 |
|---|---|---|
| FedML cross-silo + MPI 宿主 | main_fedml_shieldfl.py | ✅ |
| VeriFL 三阶段聚合 | `trainer/verifl_aggregator.py` | ✅ |
| FedAvg 基线聚合器 | `trainer/baseline_aggregator.py` | ✅ |
| GPUAccelerator（CPU fallback） | `trainer/gpu_accelerator.py` | ✅ |
| CIFAR-10 / MNIST 数据加载 + 分层均衡采样 | `data/data_loader.py` | ✅ |
| ResNet18 / ResNet20 / SimpleCNN / LeNet5 | `model/` | ✅ |
| FedMLAttacker 钩子（byzantine / model_replacement / label_flipping） | verifl_aggregator.py + `verifl_trainer.py` | ✅ |
| FedMLDefender 路径 + Bulyan 手动路径 | `baseline_aggregator.py` | ✅ |
| ASR 评估 | `eval/asr.py` | ✅ |
| 结构化指标采集 | `eval/metrics.py` | ✅ |
| 实验编排脚本 | `scripts/run_experiment.sh` | ✅ |
| cpu-deterministic 运行模式 | `utils/runtime.py` | ✅ |

### 2.2 GPU 环境硬件要求

| 项目 | 最低要求 | 推荐 |
|---|---|---|
| GPU | 1× NVIDIA GPU（Compute Capability ≥ 6.0） | 1× RTX 3090 / A100 或更高 |
| VRAM | ≥ 4 GB（ResNet18 + 5 clients 矩阵化 fitness） | ≥ 8 GB |
| CUDA | ≥ 11.7 | 12.x |
| cuDNN | 与 CUDA 版本匹配 | — |
| Driver | ≥ 515.x | 最新稳定版 |
| System RAM | ≥ 16 GB（MPI 多进程） | ≥ 32 GB |
| MPI | `mpirun` / `mpiexec` 可用 | OpenMPI 4.x |

### 2.3 软件环境要求

- Python 虚拟环境中 `fedml` 已安装且可导入
- `torch.cuda.is_available()` 返回 `True`
- `mpi4py` 已安装且 `mpirun` 可用
- CIFAR-10 / MNIST 数据已下载或可自动下载

---

## 3. GPU 运行档位定义

Phase 2 GPU 涉及两个新的运行档位。第三个（CPU）已在 Phase 2 CPU 中验证。

### 3.1 `single-gpu-deterministic`（必做）

**用途**：数值对齐验证 + 学术实验主力模式

**配置要点**：
- `using_gpu: true`
- 所有 MPI 进程（server + N clients）映射到同一张 GPU（通过 gpu_mapping.yaml）
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- `torch.use_deterministic_algorithms(True)`（若不兼容则记录例外算子并关闭该选项）
- `cpu_transfer: true`（client 模型回传先 `.cpu()` 再序列化，保证 MPI 通信不涉及 GPU tensor 直传）
- `random_seed` 固定
- client 采样顺序与聚合输入顺序固定

**预期行为**：
- 同 seed 多次运行应产生完全一致结果（或在浮点容差内一致）
- 与 CPU 运行的 MA 差异 ≤ 0.5%（前 3 轮对比）
- GPUAccelerator fitness 评估、BN 校准均在 GPU 上执行

### 3.2 `single-gpu-fast`（可选，用于大规模实验加速）

**用途**：学术规模实验的高吞吐模式

**配置要点**：
- `using_gpu: true`
- `torch.backends.cudnn.benchmark = True`
- `torch.backends.cudnn.deterministic = False`
- 不要求跨运行 bitwise 一致
- 仍要求同一实验内跨轮逻辑正确

**预期行为**：
- 比 `single-gpu-deterministic` 更快
- 不作为数值对齐基准
- 适合长轮次（50–100 rounds）正式实验

### 3.3 `multi-gpu-throughput`（条件性，取决于可用 GPU 数量）

**用途**：多 GPU 多 client 并发实验

**配置要点**：
- `using_gpu: true`
- 通过 gpu_mapping.yaml 将不同 MPI 进程映射到不同 GPU
- 不作为 bitwise 一致性基准
- 作为性能与稳定性基准

**本文档中不将此模式作为 Phase 2 GPU 完成标准**，因为其需要多 GPU 硬件。如果环境仅有单 GPU，跳过此模式不影响 Phase 2 GPU 验收。

---

## 4. 代码改动与基础设施

### 4.1 需要修改的现有文件

#### 4.1.1 `utils/runtime.py`——GPU 运行模式完善

**当前状态**：已有 `cpu-deterministic`、`single-gpu-deterministic`（seed 设置）、`single-gpu-fast` 的骨架。

**需补充的改动**：

1. 在 `configure_runtime()` 中增加 GPU 专属环境检查：

```python
if runtime_mode in ("single-gpu-deterministic", "single-gpu-fast", "multi-gpu-throughput"):
    if not torch.cuda.is_available():
        logging.error(
            "runtime_mode=%s requires CUDA but torch.cuda.is_available()=False. "
            "Falling back to cpu-deterministic.",
            runtime_mode,
        )
        runtime_mode = "cpu-deterministic"
```

2. 在 `_print_runtime_summary()` 中增加 GPU 硬件上下文输出（满足学术需求.md §3.3 硬件披露要求）：

```python
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    cuda_version = torch.version.cuda
    logging.info(
        "GPU context: name=%s vram=%.1fGB cuda=%s cudnn=%s",
        gpu_name, gpu_mem, cuda_version, torch.backends.cudnn.version(),
    )
```

3. 在 `single-gpu-deterministic` 分支中增加 `CUBLAS_WORKSPACE_CONFIG` 环境变量设置（PyTorch ≥ 1.8 要求）：

```python
if runtime_mode == "single-gpu-deterministic":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
```

#### 4.1.2 `scripts/run_experiment.sh`——支持 GPU 模式

**当前状态**：`device_args.using_gpu` 硬编码为 `false`，`runtime_mode` 硬编码为 `cpu-deterministic`。

**需补充的改动**：

1. 新增命令行参数：

```bash
GPU="false"
RUNTIME_MODE="cpu-deterministic"
GPU_MAPPING_KEY="mapping_default"
CPU_TRANSFER="true"

# 在 while 循环中增加：
--gpu)          GPU="true";           shift 1 ;;
--runtime)      RUNTIME_MODE="$2";    shift 2 ;;
--gpu_mapping)  GPU_MAPPING_KEY="$2"; shift 2 ;;
```

2. 在 YAML 模板中将硬编码替换为变量：

```yaml
device_args:
  worker_num: ${WORKER_NUM}
  using_gpu: ${GPU}
  gpu_mapping_file: config/gpu_mapping.yaml
  gpu_mapping_key: ${GPU_MAPPING_KEY}

shieldfl_args:
  runtime_mode: "${RUNTIME_MODE}"
  cpu_transfer: ${CPU_TRANSFER}
```

3. 当 `--gpu true` 时，自动将 `cpu_transfer` 设为 `true`（MPI 跨进程通信仍使用 CPU tensor）：

```bash
if [[ "$GPU" == "true" ]]; then
  CPU_TRANSFER="true"  # GPU 训练但 MPI 通信走 CPU
fi
```

#### 4.1.3 `config/gpu_mapping.yaml`——GPU 映射配置

**当前状态**：所有映射条目的 GPU ID 为 `0`（`[0, 0]`...`[0, 6]` 中第一个元素是 GPU ID）。

**需确认/补充**：

> **注意**：FedML gpu_mapping.yaml 的条目格式为 `[gpu_id, process_id]`。当前写法 `[0, 0]`, `[0, 1]`... 表示所有进程映射到 GPU 0。对于单 GPU 实验这是正确的。

新增多 GPU 映射（预留）：

```yaml
mapping_multi_gpu_5clients:
  host1:
    - [0, 0]    # server -> GPU 0
    - [0, 1]    # client 1 -> GPU 0
    - [1, 2]    # client 2 -> GPU 1
    - [0, 3]    # client 3 -> GPU 0
    - [1, 4]    # client 4 -> GPU 1

mapping_single_gpu:
  host1:
    - [0, 0]
    - [0, 1]
    - [0, 2]
    - [0, 3]
    - [0, 4]
    - [0, 5]
    - [0, 6]
    - [0, 7]
    - [0, 8]
    - [0, 9]
    - [0, 10]
```

#### 4.1.4 `trainer/gpu_accelerator.py`——GPU 路径验证日志

**当前状态**：已正确使用 `self.device` 参数，不存在硬编码 `cuda:0`。CPU fallback 已验证通过。

**需补充**：

1. 在 `__init__` 中增加 GPU 使用日志：

```python
logging.info(
    "GPUAccelerator initialized | device=%s | has_batchnorm=%s | total_params=%d | "
    "val_images_shape=%s | val_labels_shape=%s",
    self.device, self.has_batchnorm, self.total_params,
    tuple(self.val_images.shape), tuple(self.val_labels.shape),
)
```

2. 在 `set_client_parameters()` 中记录矩阵化参数的 GPU 内存占用（可选，用于硬件上下文追溯）：

```python
mem_mb = self.client_params_matrix.element_size() * self.client_params_matrix.nelement() / (1024**2)
logging.info(
    "GPUAccelerator client_params_matrix: shape=%s mem=%.1fMB device=%s",
    tuple(self.client_params_matrix.shape), mem_mb, self.device,
)
```

#### 4.1.5 `trainer/verifl_trainer.py`——移除 `cpu_transfer` 对 GPU 训练的干扰

**当前状态**：`get_model_params()` 中 `self.cpu_transfer=True` 时先 `.cpu()` 再返回 state_dict。

**确认**：在 GPU 模式下，`cpu_transfer: true` 仍应保持。因为 MPI 序列化需要 CPU tensor。训练在 GPU 上完成后，回传学习到的参数时 `.cpu()` 是正确行为。无需改动。

#### 4.1.6 `eval/metrics.py`——增加硬件上下文字段

**当前状态**：仅记录 `aggregator`, `attack_type`, `defense_type`, `model`, `dataset` 元信息。

**需补充**（满足学术需求.md §3.3）：

1. 在 `MetricsCollector.__init__()` 中采集并记录硬件上下文：

```python
self._meta["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
self._meta["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else "N/A"
self._meta["runtime_mode"] = str(getattr(args, "runtime_mode", "unknown"))
```

2. 每行 JSON 中增加 `runtime_mode` 和 `device` 字段。

### 4.2 需要新建的文件

#### 4.2.1 GPU YAML 配置文件

**新文件**：`config/fedml_config_m1_cifar10_gpu.yaml`

```yaml
# M1: 基线可信 — ResNet18 + CIFAR10, FedAvg, 无攻击无防御, 单 GPU
common_args:
  training_type: "cross_silo"
  random_seed: 0

data_args:
  dataset: "cifar10"
  data_cache_dir: ./data
  partition_method: "hetero"
  partition_alpha: 0.5
  val_per_class: 50
  trust_per_class: 50
  max_samples_per_client: 0     # 不限制（GPU 环境下不需要缩小规模）
  test_subset_size: 0           # 使用完整测试集
  num_workers: 4

model_args:
  model: "ResNet18"

train_args:
  federated_optimizer: "FedAvg"
  client_id_list:
  client_num_in_total: 10
  client_num_per_round: 10
  comm_round: 100
  epochs: 1
  batch_size: 64
  client_optimizer: sgd
  learning_rate: 0.01
  weight_decay: 0.0
  momentum: 0.9
  server_momentum: 0.9
  server_lr: 0.3
  pop_size: 15
  generations: 10
  lambda_reg: 0.01
  cpu_transfer: true
  enable_attack: false
  attack_type: "none"
  enable_defense: false
  defense_type: "none"

validation_args:
  frequency_of_the_test: 1

device_args:
  worker_num: 10
  using_gpu: true
  gpu_mapping_file: config/gpu_mapping.yaml
  gpu_mapping_key: mapping_single_gpu

comm_args:
  backend: "MPI"
  is_mobile: 0

tracking_args:
  log_file_dir: ./log
  enable_wandb: false
  using_mlops: false

shieldfl_args:
  runtime_mode: "single-gpu-deterministic"
  enforce_determinism: true
  sort_client_updates: true
  aggregator_type: "fedavg"
  metrics_output_dir: "./results"
```

**新文件**：`config/fedml_config_m1_mnist_gpu.yaml`

同上结构，仅替换：
- `dataset: "mnist"`
- `model: "LeNet5"`
- `comm_round: 50`（MNIST 收敛更快）

**新文件**：`config/fedml_config_verifl_cifar10_gpu.yaml`

同 M1 但 `aggregator_type: "verifl"`，用于 VeriFL 聚合器 GPU 验证。

#### 4.2.2 GPU 专用实验编排脚本

**新文件**：`scripts/run_m1_baseline_gpu.sh`

```bash
#!/usr/bin/env bash
# M1：基线可信 GPU 正式实验
# ResNet18 + CIFAR10 / LeNet5 + MNIST
# 10 客户端，100/50 轮，seeds={0,1,2}，alpha={0.1,0.3,0.5,100}
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

ALPHAS="0.1 0.3 0.5 100"
SEEDS="0 1 2"

echo "=== M1 Baseline GPU ==="
for ALPHA in $ALPHAS; do
  for SEED in $SEEDS; do
    echo "--- ResNet18 + CIFAR10 | alpha=${ALPHA} seed=${SEED} ---"
    bash "$DIR/run_experiment.sh" \
      --model ResNet18 --dataset cifar10 \
      --attack none --defense none --aggregator fedavg \
      --pmr 0.0 --alpha "$ALPHA" --seed "$SEED" \
      --rounds 100 --clients 10 --epochs 1 --batch_size 64 \
      --gpu --runtime single-gpu-deterministic

    echo "--- LeNet5 + MNIST | alpha=${ALPHA} seed=${SEED} ---"
    bash "$DIR/run_experiment.sh" \
      --model LeNet5 --dataset mnist \
      --attack none --defense none --aggregator fedavg \
      --pmr 0.0 --alpha "$ALPHA" --seed "$SEED" \
      --rounds 50 --clients 10 --epochs 1 --batch_size 64 \
      --gpu --runtime single-gpu-deterministic
  done
done
echo "=== M1 GPU done ==="
```

**新文件**：`scripts/run_m2_attacks_gpu.sh`

```bash
#!/usr/bin/env bash
# M2：攻击生效 GPU 正式实验
# attack × PMR × alpha × seed 全矩阵
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

ATTACKS="byzantine label_flipping model_replacement"
PMRS="0.1 0.2 0.3 0.4"
ALPHAS="0.1 0.3 0.5 100"
SEEDS="0 1 2"
CLIENTS=10

echo "=== M2 Attacks GPU ==="
for ATTACK in $ATTACKS; do
  for PMR in $PMRS; do
    for ALPHA in $ALPHAS; do
      for SEED in $SEEDS; do
        echo "--- ${ATTACK} | pmr=${PMR} alpha=${ALPHA} seed=${SEED} ---"
        bash "$DIR/run_experiment.sh" \
          --model ResNet18 --dataset cifar10 \
          --attack "$ATTACK" --defense none --aggregator fedavg \
          --pmr "$PMR" --alpha "$ALPHA" --seed "$SEED" \
          --rounds 100 --clients $CLIENTS --epochs 1 --batch_size 64 \
          --gpu --runtime single-gpu-deterministic
      done
    done
  done
done

# MNIST 线
for ATTACK in $ATTACKS; do
  for PMR in $PMRS; do
    for ALPHA in $ALPHAS; do
      for SEED in $SEEDS; do
        echo "--- ${ATTACK} + MNIST | pmr=${PMR} alpha=${ALPHA} seed=${SEED} ---"
        bash "$DIR/run_experiment.sh" \
          --model LeNet5 --dataset mnist \
          --attack "$ATTACK" --defense none --aggregator fedavg \
          --pmr "$PMR" --alpha "$ALPHA" --seed "$SEED" \
          --rounds 100 --clients $CLIENTS --epochs 1 --batch_size 64 \
          --gpu --runtime single-gpu-deterministic
      done
    done
  done
done
echo "=== M2 GPU done ==="
```

**新文件**：`scripts/run_m3_defense_gpu.sh`

```bash
#!/usr/bin/env bash
# M3：防御对照 GPU 正式实验
# defense × attack × PMR × alpha × seed 全矩阵
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

DEFENSES="RFA krum coordinate_wise_trimmed_mean cclip bulyan"
ATTACKS="byzantine label_flipping model_replacement"
PMRS="0.1 0.2 0.3 0.4"
ALPHAS="0.1 0.3 0.5 100"
SEEDS="0 1 2"
CLIENTS=10

echo "=== M3 Defense GPU (FedML 内置防御 + FedAvg) ==="
for DEFENSE in $DEFENSES; do
  for ATTACK in $ATTACKS; do
    for PMR in $PMRS; do
      for ALPHA in $ALPHAS; do
        for SEED in $SEEDS; do
          echo "--- ${DEFENSE} vs ${ATTACK} | pmr=${PMR} alpha=${ALPHA} seed=${SEED} ---"
          bash "$DIR/run_experiment.sh" \
            --model ResNet18 --dataset cifar10 \
            --attack "$ATTACK" --defense "$DEFENSE" --aggregator fedavg \
            --pmr "$PMR" --alpha "$ALPHA" --seed "$SEED" \
            --rounds 100 --clients $CLIENTS --epochs 1 --batch_size 64 \
            --gpu --runtime single-gpu-deterministic
        done
      done
    done
  done
done

echo "=== M3 Defense GPU (VeriFL 聚合器) ==="
for ATTACK in $ATTACKS; do
  for PMR in $PMRS; do
    for ALPHA in $ALPHAS; do
      for SEED in $SEEDS; do
        echo "--- VeriFL vs ${ATTACK} | pmr=${PMR} alpha=${ALPHA} seed=${SEED} ---"
        bash "$DIR/run_experiment.sh" \
          --model ResNet18 --dataset cifar10 \
          --attack "$ATTACK" --defense none --aggregator verifl \
          --pmr "$PMR" --alpha "$ALPHA" --seed "$SEED" \
          --rounds 100 --clients $CLIENTS --epochs 1 --batch_size 64 \
          --gpu --runtime single-gpu-deterministic
      done
    done
  done
done
echo "=== M3 GPU done ==="
```

**新文件**：`scripts/run_gpu_alignment.sh`

用于数值对齐验证（CPU vs GPU 对比）：

```bash
#!/usr/bin/env bash
# GPU 数值对齐验证：同 seed 下 CPU vs GPU 产出对比
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
SEED=0

echo "=== GPU Alignment: CPU baseline ==="
bash "$DIR/run_experiment.sh" \
  --model ResNet18 --dataset cifar10 \
  --attack none --defense none --aggregator fedavg \
  --pmr 0.0 --alpha 0.5 --seed $SEED \
  --rounds 5 --clients 3 --epochs 1 --batch_size 32

echo "=== GPU Alignment: GPU run ==="
bash "$DIR/run_experiment.sh" \
  --model ResNet18 --dataset cifar10 \
  --attack none --defense none --aggregator fedavg \
  --pmr 0.0 --alpha 0.5 --seed $SEED \
  --rounds 5 --clients 3 --epochs 1 --batch_size 32 \
  --gpu --runtime single-gpu-deterministic

echo "=== GPU Alignment: VeriFL CPU ==="
bash "$DIR/run_experiment.sh" \
  --model ResNet18 --dataset cifar10 \
  --attack none --defense none --aggregator verifl \
  --pmr 0.0 --alpha 0.5 --seed $SEED \
  --rounds 3 --clients 3 --epochs 1 --batch_size 32

echo "=== GPU Alignment: VeriFL GPU ==="
bash "$DIR/run_experiment.sh" \
  --model ResNet18 --dataset cifar10 \
  --attack none --defense none --aggregator verifl \
  --pmr 0.0 --alpha 0.5 --seed $SEED \
  --rounds 3 --clients 3 --epochs 1 --batch_size 32 \
  --gpu --runtime single-gpu-deterministic

echo "=== Compare results in ./results/ ==="
```

#### 4.2.3 GPU 对齐验证脚本

**新文件**：`scripts/compare_cpu_gpu_metrics.py`

Python 脚本，用于自动化对比 CPU 和 GPU 运行产出的 `.jsonl` 指标文件：

```python
"""
对比 CPU 和 GPU 运行的结构化指标文件，输出对齐报告。

用法：
  python scripts/compare_cpu_gpu_metrics.py \
    --cpu results/metrics_ResNet18_cifar10_fedavg_atknone_defnone_seed0.jsonl \
    --gpu results/metrics_ResNet18_cifar10_fedavg_atknone_defnone_seed0_gpu.jsonl \
    --tolerance 0.005
"""
import argparse
import json
import sys


def load_metrics(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compare(cpu_records, gpu_records, tolerance):
    passed = True
    for cpu_r, gpu_r in zip(cpu_records, gpu_records):
        rd = cpu_r.get("round", "?")
        cpu_acc = cpu_r.get("test_accuracy", 0)
        gpu_acc = gpu_r.get("test_accuracy", 0)
        delta = abs(cpu_acc - gpu_acc)
        status = "PASS" if delta <= tolerance else "FAIL"
        if status == "FAIL":
            passed = False
        print(f"Round {rd}: CPU_MA={cpu_acc:.4f} GPU_MA={gpu_acc:.4f} Δ={delta:.4f} [{status}]")
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--tolerance", type=float, default=0.005)
    args = parser.parse_args()

    cpu_data = load_metrics(args.cpu)
    gpu_data = load_metrics(args.gpu)
    ok = compare(cpu_data, gpu_data, args.tolerance)
    sys.exit(0 if ok else 1)
```

---

## 5. 实施步骤

### Step 1：GPU 环境预检

**目标**：确认 GPU 环境可用，FedML + MPI + CUDA 全链路就绪。

**操作**：

```bash
# 1. 确认 CUDA 可用
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 2. 确认 MPI 可用
mpirun --version

# 3. 确认 FedML 可导入
python -c "import fedml; print(fedml.__version__)"

# 4. 确认 GPU mapping 语义正确
python -c "
import torch
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem/(1024**3):.1f}GB)')
"
```

**验收点**：
- [ ] `torch.cuda.is_available()` 返回 `True`
- [ ] 至少 1 张 GPU 可用
- [ ] `mpirun` 命令可执行
- [ ] `import fedml` 成功

### Step 2：代码改动实施

按 §4.1 和 §4.2 的描述，依次完成：

1. 修改 `utils/runtime.py`：GPU 环境检查 + 硬件日志 + `CUBLAS_WORKSPACE_CONFIG`
2. 修改 `scripts/run_experiment.sh`：支持 `--gpu` / `--runtime` 参数
3. 更新 `config/gpu_mapping.yaml`：新增 `mapping_single_gpu` 配置
4. 修改 `trainer/gpu_accelerator.py`：增加 GPU 使用日志
5. 修改 `eval/metrics.py`：增加硬件上下文字段
6. 新建 GPU YAML 配置文件（至少 `m1_cifar10_gpu.yaml`、`m1_mnist_gpu.yaml`）
7. 新建 GPU 实验编排脚本
8. 新建 `scripts/compare_cpu_gpu_metrics.py`

**验收点**：
- [ ] 所有文件创建/修改完成
- [ ] `run_experiment.sh --gpu --runtime single-gpu-deterministic` 可正确生成含 `using_gpu: true` 的 YAML

### Step 3：GPU Smoke Test（最小可运行验证）

**目标**：确认 GPU 模式下 FedML + ShieldFL 全链路可执行。

**操作**：

```bash
cd python/examples/federate/prebuilt_jobs/shieldfl/

# FedAvg GPU smoke test（3 clients, 3 rounds, SimpleCNN, CIFAR10）
bash scripts/run_experiment.sh \
  --model SimpleCNN --dataset cifar10 \
  --attack none --defense none --aggregator fedavg \
  --rounds 3 --clients 3 --epochs 1 --batch_size 32 \
  --gpu --runtime single-gpu-deterministic

# VeriFL GPU smoke test（3 clients, 3 rounds, ResNet18, CIFAR10）
bash scripts/run_experiment.sh \
  --model ResNet18 --dataset cifar10 \
  --attack none --defense none --aggregator verifl \
  --rounds 3 --clients 3 --epochs 1 --batch_size 32 \
  --gpu --runtime single-gpu-deterministic
```

**验收点**：
- [ ] FedAvg GPU 运行成功，日志中显示 `using_gpu=True`
- [ ] VeriFL GPU 运行成功，日志中 GA/anchor/momentum/BN 四阶段均被触发
- [ ] `GPUAccelerator` 日志显示 `device=cuda:0`
- [ ] `results/` 下生成了 `.jsonl` 指标文件
- [ ] 指标文件中包含 `runtime_mode` 和 `device` 字段

### Step 4：CPU vs GPU 数值对齐验证

**目标**：证明 GPU 运行与 CPU 运行在同 seed 下数值对齐。

**操作**：

```bash
bash scripts/run_gpu_alignment.sh
python scripts/compare_cpu_gpu_metrics.py \
  --cpu results/metrics_ResNet18_cifar10_fedavg_atknone_defnone_seed0.jsonl \
  --gpu results/metrics_ResNet18_cifar10_fedavg_atknone_defnone_seed0.jsonl \
  --tolerance 0.005
```

> **注意**：CPU 和 GPU 的 `.jsonl` 文件名可能相同（因为 seed 相同），需要在运行时分开 `metrics_output_dir` 或手动重命名。建议在 `run_gpu_alignment.sh` 中为 CPU 和 GPU 运行分别指定不同的 `metrics_output_dir`。

**对齐判定规则**：

| 聚合器 | 对比指标 | 容差标准 | 判定方法 |
|---|---|---|---|
| FedAvg | MA（前 5 轮） | `|Δ MA| ≤ 0.5%` | 逐轮对比 |
| FedAvg | Loss（前 5 轮） | `|Δ Loss| ≤ 0.01` | 逐轮对比 |
| VeriFL | MA（前 3 轮） | `|Δ MA| ≤ 1.0%` | 逐轮对比（GA 有随机性，容差放宽） |
| VeriFL | GA best_fitness | 方向一致 | 日志手动比对 |
| VeriFL | BN 校准触发 | 必须触发 | 日志确认 |

**验收点**：
- [ ] FedAvg CPU vs GPU 对齐：前 5 轮 MA 差异 ≤ 0.5%
- [ ] VeriFL CPU vs GPU 对齐：前 3 轮 MA 差异 ≤ 1.0%
- [ ] VeriFL GPU 运行日志中 BN 校准使用 `device=cuda:0`

### Step 5：GPU Aggregation Time 基准测定

**目标**：获取 GPU 下的聚合时间数据，与 CPU 形成对比。

**操作**：

对 VeriFL 聚合器运行同配置的 CPU 和 GPU 实验（至少 3 轮），对比 `agg_time` 字段。

```bash
# CPU VeriFL
bash scripts/run_experiment.sh \
  --model ResNet18 --dataset cifar10 \
  --attack none --defense none --aggregator verifl \
  --rounds 5 --clients 5 --epochs 1 --batch_size 32

# GPU VeriFL
bash scripts/run_experiment.sh \
  --model ResNet18 --dataset cifar10 \
  --attack none --defense none --aggregator verifl \
  --rounds 5 --clients 5 --epochs 1 --batch_size 32 \
  --gpu --runtime single-gpu-deterministic
```

**预期结果**：

- GPU 下 `GPUAccelerator.calculate_fitness()` 单次调用应比 CPU 快（ResNet18 + 5 clients，预期加速比 ≥ 2×）
- 整体 `aggregate_time` 包含 GA 循环（`pop_size=15 × generations=10 = 150` 次 fitness 评估），GPU 优势更明显

**验收点**：
- [ ] 在日志或 `.jsonl` 中可提取 `agg_time`
- [ ] GPU agg_time < CPU agg_time（同配置下）
- [ ] 已记录硬件上下文（GPU 型号、VRAM、CUDA 版本）

### Step 6：M1 GPU 正式实验

**目标**：完成基线可信实验，满足学术需求.md §5 门槛。

**操作**：

```bash
bash scripts/run_m1_baseline_gpu.sh
```

**实验矩阵**：

| 任务线 | α 集合 | seed 集合 | 轮数 | 客户端数 |
|---|---|---|---|---|
| ResNet18 + CIFAR10 | {0.1, 0.3, 0.5, 100} | {0, 1, 2} | 100 | 10 |
| LeNet5 + MNIST | {0.1, 0.3, 0.5, 100} | {0, 1, 2} | 50 | 10 |

共 2 × 4 × 3 = 24 次实验。

**学术门槛验收**：

ResNet18 + CIFAR10（100 轮最终 MA，`mean ± std` over seeds）：

| α | MA 门槛 | std 门槛 |
|---|---|---|
| 100 | ≥ 85% | ≤ 2% |
| 0.5 | ≥ 82% | ≤ 2% |
| 0.3 | ≥ 78% | ≤ 2% |
| 0.1 | ≥ 75% | ≤ 2% |

LeNet5 + MNIST（50 轮最终 MA）：

| α | MA 门槛 | std 门槛 |
|---|---|---|
| 全部 | ≥ 97% | ≤ 2% |

> **阈值校准规则**：若上述门槛与实际硬件/代码环境系统性偏离（如所有 seed 在所有 α 下都差距 > 5%），允许进行一次阈值校准，但必须记录原值/新值/原因/证据实验。校准后冻结，不得在正式实验阶段反复修改。

**验收点**：
- [ ] 24 次实验全部成功完成
- [ ] 逐轮指标文件已生成
- [ ] 至少 ResNet18 + CIFAR10 α=0.5 通过 MA 门槛
- [ ] 同 seed 下 std ≤ 2%

### Step 7：M2 GPU 正式实验

**目标**：完成攻击生效实验，满足学术需求.md §6 门槛。

**操作**：

```bash
bash scripts/run_m2_attacks_gpu.sh
```

**实验矩阵**：

| 攻击 | PMR | α | seed | 任务线 | 轮数 |
|---|---|---|---|---|---|
| byzantine | {0.1, 0.2, 0.3, 0.4} | {0.1, 0.3, 0.5, 100} | {0, 1, 2} | 2 | 100 |
| label_flipping | 同上 | 同上 | 同上 | 2 | 100 |
| model_replacement | 同上 | 同上 | 同上 | 2 | 100 |

共 3 × 4 × 4 × 3 × 2 = 288 次实验。

> **建议**：先跑 PMR=20% + α∈{0.1, 0.5} 的子集验证攻击生效，再展开全矩阵。

**学术门槛验收（PMR=20%）**：

无目标攻击（byzantine / label_flipping）：
- CIFAR10：100 轮内 MA 跌破 40%
- MNIST：100 轮内 MA 跌破 50%
- 上述结论在 seeds={0,1,2} 和 α∈{0.1, 0.5} 上保持一致方向

目标后门攻击（model_replacement）：
- 隐蔽性：相对无攻击 FedAvg，MA 下降不超过 3%
- 破坏性：前 20 轮内 ASR 升至并稳定在 ≥ 85%
- 在 seeds 和 α∈{0.1, 0.5} 上保持稳定生效

**验收点**：
- [ ] 至少 PMR=20%, α∈{0.1, 0.5}, seeds={0,1,2} 的子集完成
- [ ] byzantine + CIFAR10：MA 跌破 40%（在指标文件中可验证）
- [ ] model_replacement：ASR ≥ 85% 且 MA 下降 ≤ 3%
- [ ] 结论跨 seed 方向一致

### Step 8：M3 GPU 正式实验

**目标**：完成防御对照实验，满足学术需求.md §7 门槛。

**操作**：

```bash
bash scripts/run_m3_defense_gpu.sh
```

**实验矩阵**：

FedML 内置防御（使用 `BaselineAggregator`）：
- 5 defense × 3 attack × 4 PMR × 4 α × 3 seed × 2 任务线 = 1440 次

VeriFL 聚合器（使用 `VeriFLAggregator`，FedML 内置防御关闭）：
- 1 × 3 attack × 4 PMR × 4 α × 3 seed × 2 任务线 = 288 次

**双重防御隔离检查**（必须在每次运行日志中确认）：
- VeriFL 运行时：`enable_defense: false`，日志中不出现 FedMLDefender 钩子调用
- FedML 内置防御运行时：`aggregator_type: fedavg`，日志中不出现 VeriFL 三阶段

**学术门槛验收**：
- 无攻击场景下，每种防御相对 FedAvg 的 MA 下降 ≤ 2%
- Aggregation Time 应标注 GPU 加速状态

**验收点**：
- [ ] 至少完成优先子集：krum + byzantine + PMR=20% + α=0.5 + seeds={0,1,2}
- [ ] VeriFL + byzantine 同配置完成
- [ ] 双重防御隔离在日志中可验证
- [ ] Aggregation Time 可提取并对比

### Step 9：结果汇总与交叉验证

**目标**：将全部指标汇总为可比较的表格和统计量。

**操作**：

1. 编写或使用脚本从 `results/*.jsonl` 中提取最终轮 MA/Loss/ASR：

```python
# scripts/summarize_results.py（建议新建）
# 扫描 results/ 下所有 .jsonl 文件
# 按 (model, dataset, aggregator, attack, defense, alpha, seed) 分组
# 输出 mean ± std 汇总表
```

2. 产出汇总表格式示例：

```
| Model    | Dataset | Aggregator | Attack    | Defense | α   | MA (mean±std) | ASR (mean±std) | AggTime (mean) |
|----------|---------|------------|-----------|---------|-----|---------------|----------------|-----------------|
| ResNet18 | CIFAR10 | FedAvg     | none      | none    | 0.5 | 84.2 ± 0.8%  | N/A            | 0.02s           |
| ResNet18 | CIFAR10 | VeriFL     | none      | none    | 0.5 | 83.5 ± 1.0%  | N/A            | 12.5s           |
| ResNet18 | CIFAR10 | FedAvg     | byzantine | none    | 0.5 | 35.2 ± 2.1%  | N/A            | 0.02s           |
| ResNet18 | CIFAR10 | VeriFL     | byzantine | none    | 0.5 | 78.1 ± 1.5%  | N/A            | 13.2s           |
```

**验收点**：
- [ ] 汇总表覆盖 M1/M2/M3 全部已完成实验
- [ ] 统计口径统一：`mean ± std`（over seeds）
- [ ] Aggregation Time 标注了 GPU 型号和运行模式

---

## 6. 验收标准总表

Phase 2 GPU 完成当且仅当以下全部满足：

### 6.1 环境与基础设施

- [ ] GPU 环境预检通过（CUDA + MPI + FedML 可用）
- [ ] runtime.py 已支持 GPU 环境检查和硬件日志
- [ ] run_experiment.sh 已支持 `--gpu` / `--runtime` 参数
- [ ] gpu_mapping.yaml 含单 GPU 全映射条目
- [ ] `MetricsCollector` 输出含 `runtime_mode`、`device`、GPU 型号

### 6.2 GPU Smoke Test

- [ ] FedAvg + GPU 完整运行（3 轮，无错误）
- [ ] VeriFL + GPU 完整运行（3 轮，三阶段 + BN 均触发）
- [ ] 日志确认 `GPUAccelerator.device = cuda:X`
- [ ] 日志确认 `val_images` 和 `val_labels` 在 GPU 上

### 6.3 CPU vs GPU 数值对齐

- [ ] FedAvg 对齐：前 5 轮 |Δ MA| ≤ 0.5%
- [ ] VeriFL 对齐：前 3 轮 |Δ MA| ≤ 1.0%
- [ ] VeriFL BN 校准在 GPU 上触发
- [ ] 对齐测试结果已记录（可为 Markdown 或脚本输出）

### 6.4 Aggregation Time

- [ ] GPU agg_time < CPU agg_time（VeriFL 聚合器，同配置）
- [ ] agg_time 数据可从 `.jsonl` 提取
- [ ] 已记录硬件上下文

### 6.5 M1 基线可信

- [ ] ResNet18 + CIFAR10 在 α∈{0.1, 0.3, 0.5, 100} × seeds={0,1,2} 下跑完 100 轮
- [ ] LeNet5 + MNIST 在同配置下跑完 50 轮
- [ ] 至少 α=0.5 的 ResNet18 达到 MA ≥ 82% 门槛
- [ ] LeNet5 + MNIST MA ≥ 97%
- [ ] 同配置跨 seed 标准差 ≤ 2%

### 6.6 M2 攻击生效

- [ ] 至少完成 PMR=20% + α∈{0.1, 0.5} + seeds={0,1,2} 的优先子集
- [ ] byzantine/label_flipping 使 CIFAR10 MA 跌破 40%
- [ ] model_replacement ASR ≥ 85% 且 MA 下降 ≤ 3%
- [ ] 攻击效果跨 seed 方向一致

### 6.7 M3 防御对照

- [ ] 至少完成一种内置防御（如 krum）+ 一种攻击（如 byzantine）+ VeriFL 的三方对比
- [ ] 双重防御隔离在日志中可验证
- [ ] 无攻击场景下防御 MA 下降 ≤ 2%

### 6.8 结果固化

- [ ] 所有已完成实验的 `.jsonl` 指标文件可追溯
- [ ] 汇总统计表已产出
- [ ] 失败实验已记录原因

---

## 7. GPU 特有的注意事项与已知风险

### 7.1 `torch.use_deterministic_algorithms(True)` 兼容性

部分 PyTorch 算子（如 `torch.nn.functional.interpolate`、`scatter_add_`）在 CUDA 上不支持确定性模式。如果遇到 `RuntimeError`：

- 记录具体算子和错误信息
- 将 `enforce_determinism` 设为 `false`（退回 `single-gpu-fast` 模式）
- 在验收报告中注明此例外
- 仍保留 `cudnn.deterministic=True` 和 `cudnn.benchmark=False`

### 7.2 MPI 多进程 GPU 竞争

当所有 MPI 进程映射到同一张 GPU 时（单 GPU 环境），需注意：

- 多个 client 进程的本地训练会时分复用 GPU，吞吐量受限
- Server 端 GPUAccelerator 在聚合阶段独占 GPU（此时 client 已完成本轮训练）
- 如果 VRAM 不足，减少 `num_workers` 或降低 `batch_size`

### 7.3 GPU 内存估算

| 模型 | 参数量 | 单模型 VRAM | 5 client 矩阵化 | 10 client 矩阵化 |
|---|---|---|---|---|
| SimpleCNN | ~62K | < 1 MB | < 5 MB | < 10 MB |
| LeNet5 | ~61K | < 1 MB | < 5 MB | < 10 MB |
| ResNet18 | ~11.2M | ~45 MB | ~225 MB | ~450 MB |
| ResNet20 | ~270K | ~1 MB | < 10 MB | < 20 MB |

训练时（含梯度和优化器状态），ResNet18 单进程约需 200–400 MB。10 个 client 时分复用时峰值取决于 MPI 调度。

**建议**：首次运行时通过 `nvidia-smi` 监控 GPU 内存使用，确保不 OOM。

### 7.4 `cpu_transfer` 的必要性

在 MPI backend 的 cross-silo 模式下，client 与 server 间的模型参数传输走 MPI 序列化。当前 MPI 序列化要求 CPU tensor。因此即使在 GPU 训练模式下，`cpu_transfer: true` 仍应保持：

- `VeriFLTrainer.get_model_params()` → `.cpu().state_dict()`
- `VeriFLTrainer.set_model_params()` → `load_state_dict()` 后 `model.to(device)` 在 `train()` 开头处理

这不是性能瓶颈（参数传输频率 = 每通信轮一次），但必须保持正确性。

### 7.5 MNIST 数据集下 `num_workers`

在 GPU 模式下可将 `num_workers` 提升到 2–4 以加速数据加载。但需注意：
- `num_workers > 0` 时 DataLoader 的随机性需要通过 `worker_init_fn` 和 `generator` 控制
- 当前 `_seeded_dataloader()` 已设置 `generator`，但未设置 `worker_init_fn`
- 如果在 `single-gpu-deterministic` 模式下发现多 worker 导致不确定性，退回 `num_workers: 0`

---

## 8. 实验优先级与时间分配建议

### 8.1 必做优先级（P0）

1. GPU Smoke Test（Step 3）
2. CPU vs GPU 数值对齐（Step 4）
3. M1 基线可信 GPU 实验（Step 6，至少 α=0.5 + seeds={0,1,2}）

### 8.2 高优先级（P1）

4. Aggregation Time 基准（Step 5）
5. M2 攻击生效 GPU 实验（Step 7，至少 PMR=20% 子集）
6. M3 防御对照 GPU 实验（Step 8，至少 krum 子集）

### 8.3 完整覆盖（P2）

7. M1/M2/M3 全矩阵覆盖
8. MNIST 任务线全量
9. 结果汇总与交叉验证（Step 9）
10. VeriFL Aggregation Time vs 各内置防御 Aggregation Time 对比

### 8.4 可选扩展

11. `multi-gpu-throughput` 模式验证（需多 GPU 硬件）
12. `single-gpu-fast` 模式与 `single-gpu-deterministic` 性能差异量化
13. WandB 集成验证

---

## 9. Phase 2 GPU 明确不做的事

以下不属于 Phase 2 GPU 范围：

- DDP / DataParallel 单 client 多卡训练
- cross-cloud 部署
- Launch / MLOps 全接通
- secure aggregation / DP 兼容性验证
- 新增攻击或防御算法
- ShieldFL 原有自研攻击类迁移（Phase 2 已确定使用 FedML 内置攻击）
- 改动 FedML 核心库代码
- CIFAR-100 / 其他数据集支持

---

## 10. Phase 2 GPU 最终产出

Phase 2 GPU 结束时，仓库中应新增或更新以下文件：

### 新增文件

- `config/fedml_config_m1_cifar10_gpu.yaml`
- `config/fedml_config_m1_mnist_gpu.yaml`
- `config/fedml_config_verifl_cifar10_gpu.yaml`（可选更多 GPU 配置）
- `scripts/run_m1_baseline_gpu.sh`
- `scripts/run_m2_attacks_gpu.sh`
- `scripts/run_m3_defense_gpu.sh`
- `scripts/run_gpu_alignment.sh`
- `scripts/compare_cpu_gpu_metrics.py`
- `scripts/summarize_results.py`
- `RUN_RECORD_PHASE2_GPU.md`（运行记录）

### 修改文件

- `utils/runtime.py`（GPU 环境检查 + 硬件日志）
- `scripts/run_experiment.sh`（支持 GPU 参数）
- `config/gpu_mapping.yaml`（新增 `mapping_single_gpu`）
- `trainer/gpu_accelerator.py`（GPU 使用日志）
- `eval/metrics.py`（硬件上下文字段）

### 产出数据

- `results/` 下的全部 `.jsonl` 逐轮指标文件
- 汇总统计表（Markdown 或 CSV）
- GPU vs CPU 对齐验证报告

---

## 11. 一句话版 Phase 2 GPU

> **在 Phase 2 CPU 已验证的攻防全链路上，切换到 GPU 运行模式，先做数值对齐证明迁移正确性未因设备变化而退化，再跑学术规模的 M1/M2/M3 全矩阵实验，最终产出带硬件上下文的结构化逐轮指标和可比对的汇总统计表。**
