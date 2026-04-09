#!/usr/bin/env bash
# Phase 1.1: Label Flipping 实验 (24 组)
# 实验矩阵：
#   CIFAR-10: ResNet18, 100 rounds × alpha={0.1,0.3,0.5,100} × seed={0,1,2} = 12 组
#   MNIST:    LeNet5,    50 rounds × alpha={0.1,0.3,0.5,100} × seed={0,1,2} = 12 组
#
# 攻击配置：
#   attack=label_flipping, defense=none (纯 FedAvg)
#   PMR=0.3, 映射 [0..9]→[9..0]
#   eval_asr=false (LF 不评估 ASR)
#
# 冻结参数（§6.1 + E-1）：
#   clients=10, epochs=1, batch_size=64, lr=0.01
#   max_samples_per_client=0 (无限制, 与 M1.5 baseline 一致)
#   test_subset_size=0 (完整测试集, 与 M1.5 baseline 一致)
#   momentum=0.9, server_momentum=0.0, server_lr=1.0
#
# 用法：
#   LF_GPU_ID=1 bash scripts/run_p1_lf_gpu.sh

set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

PMR="0.3"
ALPHAS="0.1 0.3 0.5 100"
SEEDS="0 1 2"
CLIENTS=10

GPU_ID="${LF_GPU_ID:-1}"

echo "=== Phase 1.1: Label Flipping GPU Experiments (24 total) ==="
echo "  GPU_ID=${GPU_ID}"
echo "  PMR=${PMR}, clients=${CLIENTS}, defense=none"
echo "  max_samples=0 (unlimited), test_subset=0 (full test set)"
echo ""

TOTAL=0
FAILED=0

echo "--- Task A: ResNet18 + CIFAR-10 (100 rounds, wd=1e-4) ---"
for ALPHA in $ALPHAS; do
	for SEED in $SEEDS; do
		TOTAL=$((TOTAL + 1))
		echo ""
		echo "[P1-LF-CIFAR10] #${TOTAL}/24 alpha=${ALPHA} seed=${SEED} pmr=${PMR}"
		echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"
		if ! bash "$DIR/run_experiment.sh" \
			--model ResNet18 --dataset cifar10 \
			--attack label_flipping --defense none \
			--pmr "${PMR}" --alpha "${ALPHA}" --seed "${SEED}" \
			--rounds 100 --clients "${CLIENTS}" --epochs 1 --batch_size 64 \
			--max_samples 0 --test_subset 0 \
			--gpu --gpu_id "${GPU_ID}" --runtime single-gpu-deterministic \
			--gpu_mapping mapping_single_gpu; then
			echo "  WARNING: experiment #${TOTAL} failed"
			FAILED=$((FAILED + 1))
		fi
		echo "  End: $(date '+%Y-%m-%d %H:%M:%S')"
	done
done

echo ""
echo "--- Task B: LeNet5 + MNIST (50 rounds, wd=0) ---"
for ALPHA in $ALPHAS; do
	for SEED in $SEEDS; do
		TOTAL=$((TOTAL + 1))
		echo ""
		echo "[P1-LF-MNIST] #${TOTAL}/24 alpha=${ALPHA} seed=${SEED} pmr=${PMR}"
		echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"
		if ! bash "$DIR/run_experiment.sh" \
			--model LeNet5 --dataset mnist \
			--attack label_flipping --defense none \
			--pmr "${PMR}" --alpha "${ALPHA}" --seed "${SEED}" \
			--rounds 50 --clients "${CLIENTS}" --epochs 1 --batch_size 64 \
			--max_samples 0 --test_subset 0 \
			--gpu --gpu_id "${GPU_ID}" --runtime single-gpu-deterministic \
			--gpu_mapping mapping_single_gpu; then
			echo "  WARNING: experiment #${TOTAL} failed"
			FAILED=$((FAILED + 1))
		fi
		echo "  End: $(date '+%Y-%m-%d %H:%M:%S')"
	done
done

echo ""
echo "=========================================="
echo "=== Phase 1.1 LF done: ${TOTAL} experiments, ${FAILED} failed ==="
echo "=========================================="
