#!/usr/bin/env bash
# M2：攻击生效 GPU 正式实验
# 实验矩阵：attack={byzantine,label_flipping,model_replacement}
#            × PMR={0.1,0.2,0.3,0.4} × alpha={0.1,0.3,0.5,100}
#            × seeds={0,1,2} × 两条任务线（CIFAR10 / MNIST）
# 共 3 × 4 × 4 × 3 × 2 = 288 次实验
# 建议先跑 PMR=20% + alpha∈{0.1,0.5} 子集验证攻击生效，再展开全矩阵
# 对应 PHASE2_GPU.md §5 Step 7
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

ATTACKS="byzantine label_flipping model_replacement"
PMRS="0.1 0.2 0.3 0.4"
ALPHAS="0.1 0.3 0.5 100"
SEEDS="0 1 2"
CLIENTS=10

echo "=== M2 Attacks GPU ==="

echo "--- ResNet18 + CIFAR10 (100 rounds) ---"
for ATTACK in $ATTACKS; do
  for PMR in $PMRS; do
    for ALPHA in $ALPHAS; do
      for SEED in $SEEDS; do
        echo "[M2-CIFAR10] attack=${ATTACK} pmr=${PMR} alpha=${ALPHA} seed=${SEED}"
        bash "$DIR/run_experiment.sh" \
          --model ResNet18 --dataset cifar10 \
          --attack "${ATTACK}" --defense none --aggregator fedavg \
          --pmr "${PMR}" --alpha "${ALPHA}" --seed "${SEED}" \
          --rounds 100 --clients "${CLIENTS}" --epochs 1 --batch_size 64 \
          --max_samples 0 --test_subset 0 \
          --gpu --gpu_id 0 --runtime single-gpu-deterministic \
          --gpu_mapping mapping_single_gpu
      done
    done
  done
done

echo "--- LeNet5 + MNIST (100 rounds) ---"
for ATTACK in $ATTACKS; do
  for PMR in $PMRS; do
    for ALPHA in $ALPHAS; do
      for SEED in $SEEDS; do
        echo "[M2-MNIST] attack=${ATTACK} pmr=${PMR} alpha=${ALPHA} seed=${SEED}"
        bash "$DIR/run_experiment.sh" \
          --model LeNet5 --dataset mnist \
          --attack "${ATTACK}" --defense none --aggregator fedavg \
          --pmr "${PMR}" --alpha "${ALPHA}" --seed "${SEED}" \
          --rounds 100 --clients "${CLIENTS}" --epochs 1 --batch_size 64 \
          --max_samples 0 --test_subset 0 \
          --gpu --gpu_id 0 --runtime single-gpu-deterministic \
          --gpu_mapping mapping_single_gpu
      done
    done
  done
done

echo "=== M2 GPU done ==="
