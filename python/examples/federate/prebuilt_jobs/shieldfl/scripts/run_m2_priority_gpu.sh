#!/usr/bin/env bash
# ================================================================
# M2 攻击实验 — GPU 优先子集
# PMR=0.2, alpha∈{0.1,0.5}, seeds={0,1,2}
# CIFAR10(50轮) + MNIST(50轮)
# 共 3 × 1 × 2 × 3 × 2 = 36 次实验
# ================================================================
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

GPU_ID="${GPU_ID:-1}" # 默认 GPU 1

ATTACKS="byzantine label_flipping model_replacement"
PMRS="0.2"
ALPHAS="0.1 0.5"
SEEDS="0 1 2"
CLIENTS=10
ROUNDS_CIFAR=50
ROUNDS_MNIST=50

TOTAL=$(echo "$ATTACKS" | wc -w)
TOTAL=$((TOTAL * 1 * 2 * 3 * 2))
COUNT=0
SKIP=0
RESULTS_DIR="$(cd "$DIR/.." && pwd)/results"

# 检查是否已有结果文件（断点续跑）
check_done() {
	local model=$1 dataset=$2 agg=$3 atk=$4 def=$5 alpha=$6 seed=$7
	local f="${RESULTS_DIR}/metrics_${model}_${dataset}_${agg}_atk${atk}_def${def}_a${alpha}_seed${seed}.jsonl"
	[[ -f "$f" ]] && return 0 || return 1
}

echo "=== M2 Attacks GPU (Priority Subset) ==="
echo "GPU_ID=${GPU_ID}  Total experiments: ${TOTAL}"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"

echo "--- ResNet18 + CIFAR10 (${ROUNDS_CIFAR} rounds) ---"
for ATTACK in $ATTACKS; do
	for PMR in $PMRS; do
		for ALPHA in $ALPHAS; do
			for SEED in $SEEDS; do
				COUNT=$((COUNT + 1))
				if check_done ResNet18 cifar10 fedavg "${ATTACK}" none "${ALPHA}" "${SEED}"; then
					echo "[$COUNT/$TOTAL] SKIP (result exists) attack=${ATTACK} alpha=${ALPHA} seed=${SEED}"
					SKIP=$((SKIP + 1))
					continue
				fi
				echo ""
				echo "[$COUNT/$TOTAL] [M2-CIFAR10] attack=${ATTACK} pmr=${PMR} alpha=${ALPHA} seed=${SEED}  $(date '+%H:%M:%S')"
				bash "$DIR/run_experiment.sh" \
					--model ResNet18 --dataset cifar10 \
					--attack "${ATTACK}" --defense none --aggregator fedavg \
					--pmr "${PMR}" --alpha "${ALPHA}" --seed "${SEED}" \
					--rounds "${ROUNDS_CIFAR}" --clients "${CLIENTS}" --epochs 1 --batch_size 64 \
					--max_samples 0 --test_subset 0 \
					--gpu --gpu_id "${GPU_ID}" --runtime single-gpu-deterministic \
					--gpu_mapping mapping_single_gpu
			done
		done
	done
done

echo "--- LeNet5 + MNIST (${ROUNDS_MNIST} rounds) ---"
for ATTACK in $ATTACKS; do
	for PMR in $PMRS; do
		for ALPHA in $ALPHAS; do
			for SEED in $SEEDS; do
				COUNT=$((COUNT + 1))
				if check_done LeNet5 mnist fedavg "${ATTACK}" none "${ALPHA}" "${SEED}"; then
					echo "[$COUNT/$TOTAL] SKIP (result exists) attack=${ATTACK} alpha=${ALPHA} seed=${SEED}"
					SKIP=$((SKIP + 1))
					continue
				fi
				echo ""
				echo "[$COUNT/$TOTAL] [M2-MNIST] attack=${ATTACK} pmr=${PMR} alpha=${ALPHA} seed=${SEED}  $(date '+%H:%M:%S')"
				bash "$DIR/run_experiment.sh" \
					--model LeNet5 --dataset mnist \
					--attack "${ATTACK}" --defense none --aggregator fedavg \
					--pmr "${PMR}" --alpha "${ALPHA}" --seed "${SEED}" \
					--rounds "${ROUNDS_MNIST}" --clients "${CLIENTS}" --epochs 1 --batch_size 64 \
					--max_samples 0 --test_subset 0 \
					--gpu --gpu_id "${GPU_ID}" --runtime single-gpu-deterministic \
					--gpu_mapping mapping_single_gpu
			done
		done
	done
done

echo ""
echo "=== M2 Priority Subset DONE at $(date '+%Y-%m-%d %H:%M:%S') (skipped ${SKIP}) ==="
