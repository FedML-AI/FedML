#!/usr/bin/env bash
# M3：防御对照 GPU 正式实验
# 两部分：
#   A) FedML 内置防御（BaselineAggregator）：
#      defense={rfa,krum,trimmed_mean,cclip}
#      × attack={byzantine,label_flipping,model_replacement}
#      × PMR={0.1,0.2,0.3,0.4} × alpha={0.1,0.3,0.5,100}
#      × seeds={0,1,2} × 两条任务线 = 5×3×4×4×3×2 = 1440 次
#   B) VeriFL 聚合器（VeriFLAggregator，内置防御关闭）：
#      × attack={byzantine,label_flipping,model_replacement}
#      × PMR={0.1,0.2,0.3,0.4} × alpha={0.1,0.3,0.5,100}
#      × seeds={0,1,2} × 两条任务线 = 3×4×4×3×2 = 288 次
# 对应 PHASE2_GPU.md §5 Step 8
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

DEFENSES="rfa krum trimmed_mean cclip"
ATTACKS="byzantine label_flipping model_replacement"
PMRS="0.1 0.2 0.3 0.4"
ALPHAS="0.1 0.3 0.5 100"
SEEDS="0 1 2"
CLIENTS=10

echo "=== M3 Defense GPU (FedML 内置防御 + FedAvg) ==="

echo "--- ResNet18 + CIFAR10 ---"
for DEFENSE in $DEFENSES; do
  for ATTACK in $ATTACKS; do
    for PMR in $PMRS; do
      for ALPHA in $ALPHAS; do
        for SEED in $SEEDS; do
          echo "[M3-FedML-CIFAR10] defense=${DEFENSE} attack=${ATTACK} pmr=${PMR} alpha=${ALPHA} seed=${SEED}"
          bash "$DIR/run_experiment.sh" \
            --model ResNet18 --dataset cifar10 \
            --attack "${ATTACK}" --defense "${DEFENSE}" --aggregator fedavg \
            --pmr "${PMR}" --alpha "${ALPHA}" --seed "${SEED}" \
            --rounds 100 --clients "${CLIENTS}" --epochs 1 --batch_size 64 \
            --max_samples 0 --test_subset 0 \
            --gpu --gpu_id 0 --runtime single-gpu-deterministic \
            --gpu_mapping mapping_single_gpu
        done
      done
    done
  done
done

echo "--- LeNet5 + MNIST ---"
for DEFENSE in $DEFENSES; do
  for ATTACK in $ATTACKS; do
    for PMR in $PMRS; do
      for ALPHA in $ALPHAS; do
        for SEED in $SEEDS; do
          echo "[M3-FedML-MNIST] defense=${DEFENSE} attack=${ATTACK} pmr=${PMR} alpha=${ALPHA} seed=${SEED}"
          bash "$DIR/run_experiment.sh" \
            --model LeNet5 --dataset mnist \
            --attack "${ATTACK}" --defense "${DEFENSE}" --aggregator fedavg \
            --pmr "${PMR}" --alpha "${ALPHA}" --seed "${SEED}" \
            --rounds 100 --clients "${CLIENTS}" --epochs 1 --batch_size 64 \
            --max_samples 0 --test_subset 0 \
            --gpu --gpu_id 0 --runtime single-gpu-deterministic \
            --gpu_mapping mapping_single_gpu
        done
      done
    done
  done
done

echo "=== M3 Defense GPU (VeriFL 聚合器，内置防御 enable_defense=false) ==="

echo "--- ResNet18 + CIFAR10 ---"
for ATTACK in $ATTACKS; do
  for PMR in $PMRS; do
    for ALPHA in $ALPHAS; do
      for SEED in $SEEDS; do
        echo "[M3-VeriFL-CIFAR10] attack=${ATTACK} pmr=${PMR} alpha=${ALPHA} seed=${SEED}"
        bash "$DIR/run_experiment.sh" \
          --model ResNet18 --dataset cifar10 \
          --attack "${ATTACK}" --defense none --aggregator verifl \
          --pmr "${PMR}" --alpha "${ALPHA}" --seed "${SEED}" \
          --rounds 100 --clients "${CLIENTS}" --epochs 1 --batch_size 64 \
          --max_samples 0 --test_subset 0 \
          --gpu --gpu_id 0 --runtime single-gpu-deterministic \
          --gpu_mapping mapping_single_gpu
      done
    done
  done
done

echo "--- LeNet5 + MNIST ---"
for ATTACK in $ATTACKS; do
  for PMR in $PMRS; do
    for ALPHA in $ALPHAS; do
      for SEED in $SEEDS; do
        echo "[M3-VeriFL-MNIST] attack=${ATTACK} pmr=${PMR} alpha=${ALPHA} seed=${SEED}"
        bash "$DIR/run_experiment.sh" \
          --model LeNet5 --dataset mnist \
          --attack "${ATTACK}" --defense none --aggregator verifl \
          --pmr "${PMR}" --alpha "${ALPHA}" --seed "${SEED}" \
          --rounds 100 --clients "${CLIENTS}" --epochs 1 --batch_size 64 \
          --max_samples 0 --test_subset 0 \
          --gpu --gpu_id 0 --runtime single-gpu-deterministic \
          --gpu_mapping mapping_single_gpu
      done
    done
  done
done

echo "=== M3 GPU done ==="
