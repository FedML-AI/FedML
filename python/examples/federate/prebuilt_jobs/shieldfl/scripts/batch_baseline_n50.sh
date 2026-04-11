#!/usr/bin/env bash
# ============================================================================
# N=50 FedAvg 无攻击基线实验批量执行脚本
# 实验矩阵：2 数据集 × 3 α × 5 seeds = 30 组
# 硬件约束：4×RTX 4090，GPU 0 被占用，使用 GPU 1,2,3
# CIFAR-10 (ResNet18): 3 GPU (1,2,3), 映射 13,19,19, T=100
# MNIST   (LeNet5)   : 2 GPU (1,2),   映射 26,25,     T=100
# 参考文档：FedAvg 无攻击基线实验设计 v2.1
# ============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

LOG_DIR="./results/batch_logs"
DONE_FILE="./results/batch_baseline_done.txt"
mkdir -p "$LOG_DIR"
touch "$DONE_FILE"

# 通用参数（§2 冻结参数）
CLIENTS=50
EPOCHS=1
BATCH=64
MAX_SAMPLES=0
TEST_SUBSET=0
LR=0.01
SERVER_LR=1.0
GPU_MAPPING=mapping_50clients_isolated

# 无攻击基线固定参数
ATTACK=none
DEFENSE=none
PMR=0.0

# CIFAR-10 参数
C10_MODEL=ResNet18
C10_DATASET=cifar10
C10_ROUNDS=100
C10_WD=0.0001
C10_GPU_ID="1,2,3"
C10_GPU_PROC="13,19,19"

# MNIST 参数（注意 T=100，与攻击实验的 T=50 不同）
MN_MODEL=LeNet5
MN_DATASET=mnist
MN_ROUNDS=100
MN_WD=0.0
MN_GPU_ID="1,2"
MN_GPU_PROC="26,25"

run_one() {
	local TAG="$1"
	local MODEL="$2"
	local DATASET="$3"
	local ALPHA="$4"
	local SEED="$5"
	local ROUNDS="$6"
	local WD="$7"
	local GPU_ID="$8"
	local GPU_PROC="$9"

	# 跳过已完成
	if grep -qF "$TAG" "$DONE_FILE" 2>/dev/null; then
		echo "[SKIP] $TAG (already done)"
		return 0
	fi

	local LOGF="$LOG_DIR/${TAG}.log"
	echo ""
	echo "================================================================"
	echo "[START] $TAG  $(date '+%Y-%m-%d %H:%M:%S')"
	echo "================================================================"

	local CMD="bash scripts/run_experiment.sh \
        --model $MODEL --dataset $DATASET \
        --attack $ATTACK --defense $DEFENSE \
        --pmr $PMR --alpha $ALPHA --seed $SEED \
        --rounds $ROUNDS --clients $CLIENTS \
        --epochs $EPOCHS --batch_size $BATCH \
        --max_samples $MAX_SAMPLES --test_subset $TEST_SUBSET \
        --gpu --gpu_id $GPU_ID \
        --runtime single-gpu-deterministic \
        --gpu_mapping $GPU_MAPPING \
        --gpu_proc_mapping $GPU_PROC \
        --lr $LR --weight_decay $WD"

	echo "[CMD] $CMD"
	local T0=$(date +%s)

	if eval "$CMD" >"$LOGF" 2>&1; then
		local T1=$(date +%s)
		local DUR=$((T1 - T0))
		echo "[DONE] $TAG  duration=${DUR}s  $(date '+%Y-%m-%d %H:%M:%S')"
		echo "$TAG  duration=${DUR}s  $(date '+%Y-%m-%d %H:%M:%S')" >>"$DONE_FILE"
	else
		local T1=$(date +%s)
		local DUR=$((T1 - T0))
		echo "[FAIL] $TAG  duration=${DUR}s  exit=$?"
		echo "[FAIL] $TAG  duration=${DUR}s" >>"$DONE_FILE"
		echo "  → see $LOGF"
		# 继续下一个实验，不中断
	fi
}

echo "========================================"
echo "  N=50 FedAvg Baseline Runner — $(date)"
echo "  Total: 30 experiments (no attack)"
echo "========================================"

# ============================================================================
# Phase 1: CIFAR-10 基线 (15 组)
# attack=none, pmr=0.0, α∈{0.1, 0.5, 100}, seed∈{0,1,2,3,4}
# ============================================================================
echo ""
echo "=== Phase 1: CIFAR-10 Baseline (15 groups) ==="

for ALPHA in 0.1 0.5 100; do
	for SEED in 0 1 2 3 4; do
		TAG="c10_baseline_a${ALPHA}_s${SEED}"
		run_one "$TAG" "$C10_MODEL" "$C10_DATASET" \
			"$ALPHA" "$SEED" \
			"$C10_ROUNDS" "$C10_WD" "$C10_GPU_ID" "$C10_GPU_PROC"
	done
done

# ============================================================================
# Phase 2: MNIST 基线 (15 组)
# attack=none, pmr=0.0, α∈{0.1, 0.5, 100}, seed∈{0,1,2,3,4}
# ============================================================================
echo ""
echo "=== Phase 2: MNIST Baseline (15 groups) ==="

for ALPHA in 0.1 0.5 100; do
	for SEED in 0 1 2 3 4; do
		TAG="mn_baseline_a${ALPHA}_s${SEED}"
		run_one "$TAG" "$MN_MODEL" "$MN_DATASET" \
			"$ALPHA" "$SEED" \
			"$MN_ROUNDS" "$MN_WD" "$MN_GPU_ID" "$MN_GPU_PROC"
	done
done

echo ""
echo "========================================"
echo "  Baseline batch complete — $(date)"
echo "  Results: $DONE_FILE"
echo "========================================"
