#!/usr/bin/env bash
# GPU 数值对齐验证：同 seed 下 CPU vs GPU 产出对比
# 分别将 CPU / GPU 结果写到隔离目录，再用 compare_cpu_gpu_metrics.py 比对
# 对应 PHASE2_GPU.md §5 Step 4
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
SEED=0

CPU_RESULTS_DIR="./results/alignment_cpu"
GPU_RESULTS_DIR="./results/alignment_gpu"

mkdir -p "$CPU_RESULTS_DIR" "$GPU_RESULTS_DIR"

# ---------- FedAvg CPU 基线（3 clients，5 轮）----------
echo "=== GPU Alignment: FedAvg CPU baseline ==="
bash "$DIR/run_experiment.sh" \
  --model ResNet18 --dataset cifar10 \
  --attack none --defense none --aggregator fedavg \
  --pmr 0.0 --alpha 0.5 --seed "${SEED}" \
  --rounds 5 --clients 3 --epochs 1 --batch_size 32 \
  --max_samples 300

# 重命名产出，避免被 GPU 运行覆盖
for f in ./results/metrics_ResNet18_cifar10_fedavg_*.jsonl; do
  [ -f "$f" ] && mv "$f" "${CPU_RESULTS_DIR}/"
done

# ---------- FedAvg GPU 运行（同配置）----------
echo "=== GPU Alignment: FedAvg GPU run ==="
bash "$DIR/run_experiment.sh" \
  --model ResNet18 --dataset cifar10 \
  --attack none --defense none --aggregator fedavg \
  --pmr 0.0 --alpha 0.5 --seed "${SEED}" \
  --rounds 5 --clients 3 --epochs 1 --batch_size 32 \
  --max_samples 300 \
  --gpu --gpu_id 0 --runtime single-gpu-deterministic \
  --gpu_mapping mapping_default

for f in ./results/metrics_ResNet18_cifar10_fedavg_*.jsonl; do
  [ -f "$f" ] && mv "$f" "${GPU_RESULTS_DIR}/"
done

# ---------- VeriFL CPU 基线（3 clients，3 轮）----------
echo "=== GPU Alignment: VeriFL CPU baseline ==="
bash "$DIR/run_experiment.sh" \
  --model ResNet18 --dataset cifar10 \
  --attack none --defense none --aggregator verifl \
  --pmr 0.0 --alpha 0.5 --seed "${SEED}" \
  --rounds 3 --clients 3 --epochs 1 --batch_size 32 \
  --max_samples 300

for f in ./results/metrics_ResNet18_cifar10_verifl_*.jsonl; do
  [ -f "$f" ] && mv "$f" "${CPU_RESULTS_DIR}/"
done

# ---------- VeriFL GPU 运行（同配置）----------
echo "=== GPU Alignment: VeriFL GPU run ==="
bash "$DIR/run_experiment.sh" \
  --model ResNet18 --dataset cifar10 \
  --attack none --defense none --aggregator verifl \
  --pmr 0.0 --alpha 0.5 --seed "${SEED}" \
  --rounds 3 --clients 3 --epochs 1 --batch_size 32 \
  --max_samples 300 \
  --gpu --gpu_id 0 --runtime single-gpu-deterministic \
  --gpu_mapping mapping_default

for f in ./results/metrics_ResNet18_cifar10_verifl_*.jsonl; do
  [ -f "$f" ] && mv "$f" "${GPU_RESULTS_DIR}/"
done

# ---------- 对齐比对 ----------
echo "=== Compare: FedAvg CPU vs GPU ==="
python scripts/compare_cpu_gpu_metrics.py \
  --cpu "${CPU_RESULTS_DIR}/metrics_ResNet18_cifar10_fedavg_atknone_defnone_seed${SEED}.jsonl" \
  --gpu "${GPU_RESULTS_DIR}/metrics_ResNet18_cifar10_fedavg_atknone_defnone_seed${SEED}.jsonl" \
  --tolerance 0.005 \
  --label "FedAvg"

echo "=== Compare: VeriFL CPU vs GPU ==="
python scripts/compare_cpu_gpu_metrics.py \
  --cpu "${CPU_RESULTS_DIR}/metrics_ResNet18_cifar10_verifl_atknone_defnone_seed${SEED}.jsonl" \
  --gpu "${GPU_RESULTS_DIR}/metrics_ResNet18_cifar10_verifl_atknone_defnone_seed${SEED}.jsonl" \
  --tolerance 0.01 \
  --label "VeriFL"

echo "=== GPU Alignment done. Results in ${CPU_RESULTS_DIR} and ${GPU_RESULTS_DIR} ==="
