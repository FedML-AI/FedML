#!/usr/bin/env bash
# M1：基线可信 GPU 正式实验（epochs=5 版本）
# 两条任务线：ResNet18 + CIFAR10（100 轮）/ LeNet5 + MNIST（50 轮）
# 实验矩阵：alpha={0.1, 0.3, 0.5, 100} × seeds={0, 1, 2}，共 24 次实验
# 对应 PHASE2_GPU.md §5 Step 6
# 变更记录：epochs 1→5，提升本地训练量以满足学术门槛
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$(cd "$DIR/.." && pwd)/results"

GPU_ID="${GPU_ID:-0}" # 默认 GPU 0，可通过环境变量覆盖
ALPHAS="0.1 0.3 0.5 100"
SEEDS="0 1 2"
CLIENTS=10
EPOCHS=5
TOTAL=24
COUNT=0

# 跳过已完成实验的检查函数
check_done() {
	local model="$1" dataset="$2" alpha="$3" seed="$4"
	local pattern="${RESULTS_DIR}/metrics_${model}_${dataset}_fedavg_atknone_defnone_a${alpha}_seed${seed}.jsonl"
	if [[ -f "$pattern" ]]; then
		return 0 # 已存在
	fi
	return 1
}

echo "=== M1 Baseline GPU (epochs=${EPOCHS}) ==="
echo "GPU_ID=${GPU_ID}  Total experiments: ${TOTAL}"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "--- ResNet18 + CIFAR10 (100 rounds) ---"
for ALPHA in $ALPHAS; do
	for SEED in $SEEDS; do
		COUNT=$((COUNT + 1))
		if check_done ResNet18 cifar10 "${ALPHA}" "${SEED}"; then
			echo "[$COUNT/$TOTAL] SKIP (result exists) alpha=${ALPHA} seed=${SEED}"
			continue
		fi
		echo ""
		echo "[$COUNT/$TOTAL] [M1-CIFAR10] alpha=${ALPHA} seed=${SEED}  $(date '+%H:%M:%S')"
		bash "$DIR/run_experiment.sh" \
			--model ResNet18 --dataset cifar10 \
			--attack none --defense none --aggregator fedavg \
			--pmr 0.0 --alpha "${ALPHA}" --seed "${SEED}" \
			--rounds 100 --clients "${CLIENTS}" --epochs "${EPOCHS}" --batch_size 64 \
			--max_samples 0 --test_subset 0 \
			--gpu --gpu_id "${GPU_ID}" --runtime single-gpu-deterministic \
			--gpu_mapping mapping_single_gpu \
		|| echo "WARNING: experiment failed, continuing..."
	done
done

echo "--- LeNet5 + MNIST (50 rounds) ---"
for ALPHA in $ALPHAS; do
	for SEED in $SEEDS; do
		COUNT=$((COUNT + 1))
		if check_done LeNet5 mnist "${ALPHA}" "${SEED}"; then
			echo "[$COUNT/$TOTAL] SKIP (result exists) alpha=${ALPHA} seed=${SEED}"
			continue
		fi
		echo ""
		echo "[$COUNT/$TOTAL] [M1-MNIST] alpha=${ALPHA} seed=${SEED}  $(date '+%H:%M:%S')"
		bash "$DIR/run_experiment.sh" \
			--model LeNet5 --dataset mnist \
			--attack none --defense none --aggregator fedavg \
			--pmr 0.0 --alpha "${ALPHA}" --seed "${SEED}" \
			--rounds 50 --clients "${CLIENTS}" --epochs "${EPOCHS}" --batch_size 64 \
			--max_samples 0 --test_subset 0 \
			--gpu --gpu_id "${GPU_ID}" --runtime single-gpu-deterministic \
			--gpu_mapping mapping_single_gpu \
		|| echo "WARNING: experiment failed, continuing..."
	done
done

echo ""
echo "=== M1 GPU DONE at $(date '+%Y-%m-%d %H:%M:%S') ==="
