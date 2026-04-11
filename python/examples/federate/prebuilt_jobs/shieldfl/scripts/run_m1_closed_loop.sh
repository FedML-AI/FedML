#!/usr/bin/env bash
# M1 闭环学术实验：严格按照《M1 闭环学术需求.md》执行
# ===== 冻结参数配方 =====
# local_epochs: 1
# batch_size: 64
# client_optimizer: sgd, momentum=0.9
# learning_rate: 0.01 (both tasks)
# weight_decay: 1e-4 (CIFAR-10), 0 (MNIST)
# server_lr: 1.0 (pure FedAvg, η_g=1.0)
# client_num_in_total: 10, client_num_per_round: 10 (full participation)
# CIFAR-10: 100 rounds, MNIST: 50 rounds
# α = {0.1, 0.3, 0.5, 100}, seeds = {0,1,2} (α=0.1 CIFAR-10: seeds={0,1,2,3,4})
# attack: none, defense: none, aggregator: fedavg
# ===== 实验矩阵 =====
# CIFAR-10: 5 + 3×3 = 14 experiments
# MNIST:    4×3     = 12 experiments
# Total:              26 experiments
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$(cd "$DIR/.." && pwd)/results"

GPU_ID="${GPU_ID:-0}"
CLIENTS=10
EPOCHS=1
BATCH=64
LR="0.01"
SERVER_LR="1.0"

check_done() {
	local model="$1" dataset="$2" alpha="$3" seed="$4"
	local pattern="${RESULTS_DIR}/metrics_${model}_${dataset}_fedavg_atknone_defnone_a${alpha}_seed${seed}.jsonl"
	if [[ -f "$pattern" ]]; then
		# 检查文件是否有足够的行数（至少有 rounds/2 行数据说明基本完成）
		local lines
		lines=$(wc -l <"$pattern")
		if [[ "$lines" -ge 10 ]]; then
			return 0
		fi
	fi
	return 1
}

run_one() {
	local model="$1" dataset="$2" rounds="$3" alpha="$4" seed="$5" wd="$6"
	if check_done "$model" "$dataset" "$alpha" "$seed"; then
		echo "  SKIP (result exists) ${model} ${dataset} α=${alpha} seed=${seed}"
		return 0
	fi
	echo ""
	echo "  [${model}+${dataset}] α=${alpha} seed=${seed} rounds=${rounds} wd=${wd}  $(date '+%H:%M:%S')"
	bash "$DIR/run_experiment.sh" \
		--model "$model" --dataset "$dataset" \
		--attack none --defense none --aggregator fedavg \
		--pmr 0.0 --alpha "$alpha" --seed "$seed" \
		--rounds "$rounds" --clients "${CLIENTS}" --epochs "${EPOCHS}" --batch_size "${BATCH}" \
		--lr "${LR}" --server_lr "${SERVER_LR}" --weight_decay "$wd" \
		--max_samples 0 --test_subset 0 \
		--gpu --gpu_id "${GPU_ID}" --runtime single-gpu-deterministic \
		--gpu_mapping mapping_single_gpu ||
		echo "WARNING: experiment failed (${model} ${dataset} α=${alpha} seed=${seed})"
}

TOTAL_C=14 # CIFAR-10: 5 (α=0.1) + 9 (α=0.3/0.5/100)
TOTAL_M=12 # MNIST: 4α × 3seeds
TOTAL=$((TOTAL_C + TOTAL_M))
COUNT=0

echo "=========================================="
echo " M1 Closed-Loop Academic Experiments"
echo " Frozen recipe: E=${EPOCHS} lr=${LR} bs=${BATCH} server_lr=${SERVER_LR}"
echo " GPU_ID=${GPU_ID}  Total experiments: ${TOTAL}"
echo " Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# ===== Task A: ResNet18 + CIFAR10, 100 rounds =====
echo ""
echo "=== Task A: ResNet18 + CIFAR10 (100 rounds, wd=1e-4) ==="

# α=0.1: 5 seeds (known high std from prior data)
for SEED in 0 1 2 3 4; do
	COUNT=$((COUNT + 1))
	echo "[$COUNT/$TOTAL]"
	run_one ResNet18 cifar10 100 0.1 "$SEED" "0.0001"
done

# α=0.3, 0.5, 100: 3 seeds each
for ALPHA in 0.3 0.5 100; do
	for SEED in 0 1 2; do
		COUNT=$((COUNT + 1))
		echo "[$COUNT/$TOTAL]"
		run_one ResNet18 cifar10 100 "$ALPHA" "$SEED" "0.0001"
	done
done

# ===== Task B: LeNet5 + MNIST, 50 rounds =====
echo ""
echo "=== Task B: LeNet5 + MNIST (50 rounds, wd=0) ==="

for ALPHA in 0.1 0.3 0.5 100; do
	for SEED in 0 1 2; do
		COUNT=$((COUNT + 1))
		echo "[$COUNT/$TOTAL]"
		run_one LeNet5 mnist 50 "$ALPHA" "$SEED" "0.0"
	done
done

echo ""
echo "=========================================="
echo " M1 Closed-Loop DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
