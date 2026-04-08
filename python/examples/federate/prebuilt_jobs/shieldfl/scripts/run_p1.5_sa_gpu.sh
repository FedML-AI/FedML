#!/usr/bin/env bash
# P1.5: Scaling Attack 验证实验 (35 组)
# 基于 M2_SA_FIX_PHASE1_ERROR_FIX.md 规格
#
# 变更项（vs P1）：
#   D-P1.5-1: 攻击窗口从末段5轮改为单轮 → CIFAR-10 [99], MNIST [49]
#   D-P1.5-2: γ 从固定10改为 auto (γ = Σn_i / n_malicious)
#   D-P1.5-3: MNIST 增加 seed=3,4 共5个seed
#
# 实验矩阵：
#   CIFAR-10 主实验 12 组 (γ=auto, attack=[99]):
#     ResNet18, 100 rounds × α={0.1,0.3,0.5,100} × seed={0,1,2}
#   MNIST 主实验 20 组 (γ=auto, attack=[49]):
#     LeNet5, 50 rounds × α={0.1,0.3,0.5,100} × seed={0,1,2,3,4}
#   控制实验 3 组 (γ=1, 单轮):
#     CIFAR-10 α=0.5 seed=0, CIFAR-10 α=100 seed=0, MNIST α=0.5 seed=0
#   总计：35 组
#
# 冻结参数：
#   PMR=0.1 → K=1, defense=none (纯 FedAvg)
#   clients=10, epochs=1, batch_size=64, lr=0.01
#   max_samples=0, test_subset=0
#   backdoor_per_batch=20
#
# 用法：
#   SA_GPU_ID=2 bash scripts/run_p1.5_sa_gpu.sh

set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

PMR="0.1"
ALPHAS="0.1 0.3 0.5 100"
SEEDS_3="0 1 2"
SEEDS_5="0 1 2 3 4"
CLIENTS=10

GPU_ID="${SA_GPU_ID:-2}"

echo "=== P1.5: Scaling Attack Verification (35 total) ==="
echo "  GPU_ID=${GPU_ID}"
echo "  PMR=${PMR} → K=1, defense=none"
echo "  D-P1.5-1: single-round attack (CIFAR-10 [99], MNIST [49])"
echo "  D-P1.5-2: scale_gamma=auto"
echo "  D-P1.5-3: MNIST 5 seeds"
echo ""

TOTAL=0
FAILED=0

# =========================================================
# Task A: CIFAR-10 主实验 (12 组, γ=auto, attack=[99])
# =========================================================
echo "--- Task A: ResNet18 + CIFAR-10 (100 rounds, γ=auto, attack=[99]) ---"
for ALPHA in $ALPHAS; do
	for SEED in $SEEDS_3; do
		TOTAL=$((TOTAL + 1))
		echo ""
		echo "[P1.5-SA-CIFAR10] #${TOTAL}/35 alpha=${ALPHA} seed=${SEED} γ=auto attack=[99]"
		echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"
		if ! bash "$DIR/run_experiment.sh" \
			--model ResNet18 --dataset cifar10 \
			--attack model_replacement --defense none \
			--pmr "${PMR}" --alpha "${ALPHA}" --seed "${SEED}" \
			--rounds 100 --clients "${CLIENTS}" --epochs 1 --batch_size 64 \
			--max_samples 0 --test_subset 0 \
			--scale_gamma auto \
			--attack_rounds "[99]" \
			--gpu --gpu_id "${GPU_ID}" --runtime single-gpu-deterministic \
			--gpu_mapping mapping_single_gpu; then
			echo "  WARNING: experiment #${TOTAL} FAILED"
			FAILED=$((FAILED + 1))
		fi
		echo "  End: $(date '+%Y-%m-%d %H:%M:%S')"
	done
done

# =========================================================
# Task B: MNIST 主实验 (20 组, γ=auto, attack=[49])
# =========================================================
echo ""
echo "--- Task B: LeNet5 + MNIST (50 rounds, γ=auto, attack=[49]) ---"
for ALPHA in $ALPHAS; do
	for SEED in $SEEDS_5; do
		TOTAL=$((TOTAL + 1))
		echo ""
		echo "[P1.5-SA-MNIST] #${TOTAL}/35 alpha=${ALPHA} seed=${SEED} γ=auto attack=[49]"
		echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"
		if ! bash "$DIR/run_experiment.sh" \
			--model LeNet5 --dataset mnist \
			--attack model_replacement --defense none \
			--pmr "${PMR}" --alpha "${ALPHA}" --seed "${SEED}" \
			--rounds 50 --clients "${CLIENTS}" --epochs 1 --batch_size 64 \
			--max_samples 0 --test_subset 0 \
			--scale_gamma auto \
			--attack_rounds "[49]" \
			--gpu --gpu_id "${GPU_ID}" --runtime single-gpu-deterministic \
			--gpu_mapping mapping_single_gpu; then
			echo "  WARNING: experiment #${TOTAL} FAILED"
			FAILED=$((FAILED + 1))
		fi
		echo "  End: $(date '+%Y-%m-%d %H:%M:%S')"
	done
done

# =========================================================
# Archive main γ=auto results that collide with γ=1 control
# =========================================================
RESULTS_DIR="${DIR}/../results"
GAMMA_AUTO_ARCHIVE="${RESULTS_DIR}/p1.5_gamma_auto_ctrl_backup"
mkdir -p "${GAMMA_AUTO_ARCHIVE}"

echo ""
echo "--- Archiving γ=auto results that overlap with γ=1 control ---"

# CIFAR-10 α=0.5 seed=0
SRC="${RESULTS_DIR}/metrics_ResNet18_cifar10_shieldfl_atkmodel_replacement_defnone_a0.5_pmr${PMR}_seed0.jsonl"
if [[ -f "$SRC" ]]; then
	cp "$SRC" "${GAMMA_AUTO_ARCHIVE}/"
	echo "  Backed up: $(basename "$SRC")"
fi

# CIFAR-10 α=100 seed=0
SRC="${RESULTS_DIR}/metrics_ResNet18_cifar10_shieldfl_atkmodel_replacement_defnone_a100_pmr${PMR}_seed0.jsonl"
if [[ -f "$SRC" ]]; then
	cp "$SRC" "${GAMMA_AUTO_ARCHIVE}/"
	echo "  Backed up: $(basename "$SRC")"
fi

# MNIST α=0.5 seed=0
SRC="${RESULTS_DIR}/metrics_LeNet5_mnist_shieldfl_atkmodel_replacement_defnone_a0.5_pmr${PMR}_seed0.jsonl"
if [[ -f "$SRC" ]]; then
	cp "$SRC" "${GAMMA_AUTO_ARCHIVE}/"
	echo "  Backed up: $(basename "$SRC")"
fi

# =========================================================
# Task C: 控制实验 (3 组, γ=1, 单轮)
# =========================================================
echo ""
echo "--- Task C: Control group (γ=1, single-round) ---"

# Control 1: CIFAR-10 α=0.5 seed=0 γ=1
TOTAL=$((TOTAL + 1))
echo ""
echo "[P1.5-SA-CTRL] #${TOTAL}/35 CIFAR-10 alpha=0.5 seed=0 γ=1 attack=[99]"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"
if ! bash "$DIR/run_experiment.sh" \
	--model ResNet18 --dataset cifar10 \
	--attack model_replacement --defense none \
	--pmr "${PMR}" --alpha "0.5" --seed "0" \
	--rounds 100 --clients "${CLIENTS}" --epochs 1 --batch_size 64 \
	--max_samples 0 --test_subset 0 \
	--scale_gamma 1 \
	--attack_rounds "[99]" \
	--gpu --gpu_id "${GPU_ID}" --runtime single-gpu-deterministic \
	--gpu_mapping mapping_single_gpu; then
	echo "  WARNING: experiment #${TOTAL} FAILED"
	FAILED=$((FAILED + 1))
fi
echo "  End: $(date '+%Y-%m-%d %H:%M:%S')"

# Rename γ=1 control to avoid future collision
CTRL_DIR="${RESULTS_DIR}/p1.5_gamma1_ctrl"
mkdir -p "${CTRL_DIR}"
SRC="${RESULTS_DIR}/metrics_ResNet18_cifar10_shieldfl_atkmodel_replacement_defnone_a0.5_pmr${PMR}_seed0.jsonl"
if [[ -f "$SRC" ]]; then
	mv "$SRC" "${CTRL_DIR}/metrics_ResNet18_cifar10_ctrl_g1_a0.5_seed0.jsonl"
	echo "  Moved γ=1 result → p1.5_gamma1_ctrl/"
fi

# Restore γ=auto result
BACKUP="${GAMMA_AUTO_ARCHIVE}/metrics_ResNet18_cifar10_shieldfl_atkmodel_replacement_defnone_a0.5_pmr${PMR}_seed0.jsonl"
if [[ -f "$BACKUP" ]]; then
	cp "$BACKUP" "${RESULTS_DIR}/"
	echo "  Restored γ=auto result"
fi

# Control 2: CIFAR-10 α=100 seed=0 γ=1
TOTAL=$((TOTAL + 1))
echo ""
echo "[P1.5-SA-CTRL] #${TOTAL}/35 CIFAR-10 alpha=100 seed=0 γ=1 attack=[99]"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"
if ! bash "$DIR/run_experiment.sh" \
	--model ResNet18 --dataset cifar10 \
	--attack model_replacement --defense none \
	--pmr "${PMR}" --alpha "100" --seed "0" \
	--rounds 100 --clients "${CLIENTS}" --epochs 1 --batch_size 64 \
	--max_samples 0 --test_subset 0 \
	--scale_gamma 1 \
	--attack_rounds "[99]" \
	--gpu --gpu_id "${GPU_ID}" --runtime single-gpu-deterministic \
	--gpu_mapping mapping_single_gpu; then
	echo "  WARNING: experiment #${TOTAL} FAILED"
	FAILED=$((FAILED + 1))
fi
echo "  End: $(date '+%Y-%m-%d %H:%M:%S')"

# Rename γ=1 control
SRC="${RESULTS_DIR}/metrics_ResNet18_cifar10_shieldfl_atkmodel_replacement_defnone_a100_pmr${PMR}_seed0.jsonl"
if [[ -f "$SRC" ]]; then
	mv "$SRC" "${CTRL_DIR}/metrics_ResNet18_cifar10_ctrl_g1_a100_seed0.jsonl"
	echo "  Moved γ=1 result → p1.5_gamma1_ctrl/"
fi

# Restore γ=auto result
BACKUP="${GAMMA_AUTO_ARCHIVE}/metrics_ResNet18_cifar10_shieldfl_atkmodel_replacement_defnone_a100_pmr${PMR}_seed0.jsonl"
if [[ -f "$BACKUP" ]]; then
	cp "$BACKUP" "${RESULTS_DIR}/"
	echo "  Restored γ=auto result"
fi

# Control 3: MNIST α=0.5 seed=0 γ=1
TOTAL=$((TOTAL + 1))
echo ""
echo "[P1.5-SA-CTRL] #${TOTAL}/35 MNIST alpha=0.5 seed=0 γ=1 attack=[49]"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"
if ! bash "$DIR/run_experiment.sh" \
	--model LeNet5 --dataset mnist \
	--attack model_replacement --defense none \
	--pmr "${PMR}" --alpha "0.5" --seed "0" \
	--rounds 50 --clients "${CLIENTS}" --epochs 1 --batch_size 64 \
	--max_samples 0 --test_subset 0 \
	--scale_gamma 1 \
	--attack_rounds "[49]" \
	--gpu --gpu_id "${GPU_ID}" --runtime single-gpu-deterministic \
	--gpu_mapping mapping_single_gpu; then
	echo "  WARNING: experiment #${TOTAL} FAILED"
	FAILED=$((FAILED + 1))
fi
echo "  End: $(date '+%Y-%m-%d %H:%M:%S')"

# Rename γ=1 control
SRC="${RESULTS_DIR}/metrics_LeNet5_mnist_shieldfl_atkmodel_replacement_defnone_a0.5_pmr${PMR}_seed0.jsonl"
if [[ -f "$SRC" ]]; then
	mv "$SRC" "${CTRL_DIR}/metrics_LeNet5_mnist_ctrl_g1_a0.5_seed0.jsonl"
	echo "  Moved γ=1 result → p1.5_gamma1_ctrl/"
fi

# Restore γ=auto result
BACKUP="${GAMMA_AUTO_ARCHIVE}/metrics_LeNet5_mnist_shieldfl_atkmodel_replacement_defnone_a0.5_pmr${PMR}_seed0.jsonl"
if [[ -f "$BACKUP" ]]; then
	cp "$BACKUP" "${RESULTS_DIR}/"
	echo "  Restored γ=auto result"
fi

# =========================================================
# Summary
# =========================================================
echo ""
echo "=========================================="
echo "=== P1.5 SA done: ${TOTAL} experiments, ${FAILED} failed ==="
echo "=========================================="
echo ""
echo "Results layout:"
echo "  Main experiments: ${RESULTS_DIR}/*.jsonl"
echo "  γ=1 control:     ${CTRL_DIR}/"
echo "  γ=auto backup:   ${GAMMA_AUTO_ARCHIVE}/"
