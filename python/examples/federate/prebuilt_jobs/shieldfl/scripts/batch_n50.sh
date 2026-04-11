#!/usr/bin/env bash
# ============================================================================
# N=50 正式实验批量执行脚本
# 实验矩阵：24 主实验 + 4 控制组 = 28 组
# 硬件约束：4×RTX 4090，GPU 0 被占用，使用 GPU 1,2,3
# CIFAR-10 (ResNet18): 3 GPU, 映射 13,19,19 (~100 轮，~50 min/exp)
# MNIST   (LeNet5)   : 2 GPU, 映射 26,25     (~50 轮,  ~20 min/exp)
# ============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

LOG_DIR="./results/batch_logs"
DONE_FILE="./results/batch_done.txt"
mkdir -p "$LOG_DIR"
touch "$DONE_FILE"

# 通用参数
CLIENTS=50
EPOCHS=1
BATCH=64
MAX_SAMPLES=0
TEST_SUBSET=0
LR=0.01
SERVER_LR=1.0
BACKDOOR_PER_BATCH=20
GPU_MAPPING=mapping_50clients_isolated

# CIFAR-10 参数
C10_MODEL=ResNet18
C10_DATASET=cifar10
C10_ROUNDS=100
C10_WD=0.0001
C10_GPU_ID="1,2,3"
C10_GPU_PROC="13,19,19"

# MNIST 参数
MN_MODEL=LeNet5
MN_DATASET=mnist
MN_ROUNDS=50
MN_WD=0.0
MN_GPU_ID="1,2"
MN_GPU_PROC="26,25"

run_one() {
	local TAG="$1"
	local MODEL="$2"
	local DATASET="$3"
	local ATTACK="$4"
	local DEFENSE="$5"
	local PMR="$6"
	local ALPHA="$7"
	local SEED="$8"
	local ROUNDS="$9"
	local WD="${10}"
	local GPU_ID="${11}"
	local GPU_PROC="${12}"
	local EXTRA_ARGS="${13:-}"

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
        --lr $LR --weight_decay $WD \
        --scale_gamma auto \
        --backdoor_per_batch $BACKDOOR_PER_BATCH \
        $EXTRA_ARGS"

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
echo "  N=50 Batch Runner — $(date)"
echo "  Total: 28 experiments"
echo "========================================"

# ============================================================================
# Phase 1: CIFAR-10 主实验 (9 组)
# PMR=20%, α∈{0.1, 0.5, 100}, seed∈{0,1,2}
# ============================================================================
echo ""
echo "=== Phase 1: CIFAR-10 Main Experiments (9 groups) ==="

for ALPHA in 0.1 0.5 100; do
	for SEED in 0 1 2; do
		TAG="c10_atk_a${ALPHA}_s${SEED}"
		run_one "$TAG" "$C10_MODEL" "$C10_DATASET" \
			model_replacement none 0.2 "$ALPHA" "$SEED" \
			"$C10_ROUNDS" "$C10_WD" "$C10_GPU_ID" "$C10_GPU_PROC" ""
	done
done

# ============================================================================
# Phase 2: MNIST 主实验 (15 组)
# PMR=20%, α∈{0.1, 0.5, 100}, seed∈{0,1,2,3,4}
# ============================================================================
echo ""
echo "=== Phase 2: MNIST Main Experiments (15 groups) ==="

for ALPHA in 0.1 0.5 100; do
	for SEED in 0 1 2 3 4; do
		TAG="mn_atk_a${ALPHA}_s${SEED}"
		run_one "$TAG" "$MN_MODEL" "$MN_DATASET" \
			model_replacement none 0.2 "$ALPHA" "$SEED" \
			"$MN_ROUNDS" "$MN_WD" "$MN_GPU_ID" "$MN_GPU_PROC" ""
	done
done

# ============================================================================
# Phase 3: 控制组 (4 组)
# ============================================================================
echo ""
echo "=== Phase 3: Control Groups (4 groups) ==="

# γ=1 因果性控制 (PMR=20%, α=100, seed=0, scale_gamma=1)
TAG="c10_ctrl_gamma1"
run_one "$TAG" "$C10_MODEL" "$C10_DATASET" \
	model_replacement none 0.2 100 0 \
	"$C10_ROUNDS" "$C10_WD" "$C10_GPU_ID" "$C10_GPU_PROC" \
	"--scale_gamma 1"

TAG="mn_ctrl_gamma1"
run_one "$TAG" "$MN_MODEL" "$MN_DATASET" \
	model_replacement none 0.2 100 0 \
	"$MN_ROUNDS" "$MN_WD" "$MN_GPU_ID" "$MN_GPU_PROC" \
	"--scale_gamma 1"

# 无攻击 FedAvg 基线 (PMR=0%, α=0.5, seed=0)
TAG="c10_baseline_noatk"
run_one "$TAG" "$C10_MODEL" "$C10_DATASET" \
	none none 0.0 0.5 0 \
	"$C10_ROUNDS" "$C10_WD" "$C10_GPU_ID" "$C10_GPU_PROC" ""

TAG="mn_baseline_noatk"
run_one "$TAG" "$MN_MODEL" "$MN_DATASET" \
	none none 0.0 0.5 0 \
	"$MN_ROUNDS" "$MN_WD" "$MN_GPU_ID" "$MN_GPU_PROC" ""

echo ""
echo "========================================"
echo "  Batch complete — $(date)"
echo "  Results: $DONE_FILE"
echo "========================================"
