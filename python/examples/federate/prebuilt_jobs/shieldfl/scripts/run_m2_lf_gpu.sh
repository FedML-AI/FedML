#!/usr/bin/env bash
# M2 Label Flipping 专项实验 (24组)
# 实验矩阵：
#   CIFAR-10: ResNet18, 100 rounds × alpha={0.1,0.3,0.5,100} × seed={0,1,2} = 12 组
#   MNIST:    LeNet5,    50 rounds × alpha={0.1,0.3,0.5,100} × seed={0,1,2} = 12 组
# 攻击配置：PMR=0.3, 全程投毒, 标签映射 [0..9]->[9..0]
# 配方对齐 LF_实施规格.md §7
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

PMR="0.3"
ALPHAS="0.1 0.3 0.5 100"
SEEDS="0 1 2"
CLIENTS=10

# 默认 GPU 参数
GPU_ID="${LF_GPU_ID:-0}"

echo "=== M2 Label Flipping GPU Experiments (24 total) ==="
echo "  GPU_ID=${GPU_ID}"

TOTAL=0
FAILED=0

echo ""
echo "--- Task A: ResNet18 + CIFAR-10 (100 rounds, wd=1e-4) ---"
for ALPHA in $ALPHAS; do
	for SEED in $SEEDS; do
		TOTAL=$((TOTAL + 1))
		echo "[LF-CIFAR10] #${TOTAL} alpha=${ALPHA} seed=${SEED} pmr=${PMR}"
		if ! bash "$DIR/run_experiment.sh" \
			--model ResNet18 --dataset cifar10 \
			--attack label_flipping --defense none --aggregator fedavg \
			--pmr "${PMR}" --alpha "${ALPHA}" --seed "${SEED}" \
			--rounds 100 --clients "${CLIENTS}" --epochs 1 --batch_size 64 \
			--max_samples 0 --test_subset 0 \
			--gpu --gpu_id "${GPU_ID}" --runtime single-gpu-deterministic \
			--gpu_mapping mapping_single_gpu; then
			echo "  WARNING: experiment #${TOTAL} failed"
			FAILED=$((FAILED + 1))
		fi
	done
done

echo ""
echo "--- Task B: LeNet5 + MNIST (50 rounds, wd=0) ---"
for ALPHA in $ALPHAS; do
	for SEED in $SEEDS; do
		TOTAL=$((TOTAL + 1))
		echo "[LF-MNIST] #${TOTAL} alpha=${ALPHA} seed=${SEED} pmr=${PMR}"
		if ! bash "$DIR/run_experiment.sh" \
			--model LeNet5 --dataset mnist \
			--attack label_flipping --defense none --aggregator fedavg \
			--pmr "${PMR}" --alpha "${ALPHA}" --seed "${SEED}" \
			--rounds 50 --clients "${CLIENTS}" --epochs 1 --batch_size 64 \
			--max_samples 0 --test_subset 0 \
			--gpu --gpu_id "${GPU_ID}" --runtime single-gpu-deterministic \
			--gpu_mapping mapping_single_gpu; then
			echo "  WARNING: experiment #${TOTAL} failed"
			FAILED=$((FAILED + 1))
		fi
	done
done

echo ""
echo "=== M2 LF GPU done: ${TOTAL} experiments, ${FAILED} failed ==="
