#!/usr/bin/env bash
# Phase 1.2: Scaling Attack (Model Replacement) 实验 (27 组)
# 实验矩阵：
#   正式实验 24 组 (γ=10):
#     CIFAR-10: ResNet18, 100 rounds × alpha={0.1,0.3,0.5,100} × seed={0,1,2} = 12 组
#     MNIST:    LeNet5,    50 rounds × alpha={0.1,0.3,0.5,100} × seed={0,1,2} = 12 组
#   控制实验 3 组 (γ=1):
#     CIFAR-10: ResNet18, 100 rounds × alpha=0.5 × seed={0,1,2} = 3 组
#
# 攻击配置（关键修订 vs M2）：
#   attack=model_replacement, defense=none (纯 FedAvg)
#   PMR=0.1 → K=ceil(10×0.1)=1 (恢复单攻击者语义)
#   γ=10 (默认), γ=1 (控制组)
#   eval_asr=true
#   target_label=0, trigger_size=3, trigger_value=1.0
#   backdoor_per_batch=20, 末段 5 轮攻击窗口
#
# 数学验证 (K=1, γ=10, N=10, max_samples=300 → 等权 FedAvg):
#   G^{t+1} = (1/10)[10(W_m - G) + G + 9G] = W_m (精确 model replacement)
#
# 冻结参数（§6.1 + E-1）：
#   clients=10, epochs=1, batch_size=64, lr=0.01
#   max_samples_per_client=0 (无限制, 与 M1.5 baseline 一致)
#   test_subset_size=0 (完整测试集, 与 M1.5 baseline 一致)
#   momentum=0.9, server_momentum=0.0, server_lr=1.0
#
# 用法：
#   SA_GPU_ID=2 bash scripts/run_p1_sa_gpu.sh

set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

PMR="0.1"
ALPHAS="0.1 0.3 0.5 100"
SEEDS="0 1 2"
CLIENTS=10

GPU_ID="${SA_GPU_ID:-2}"

echo "=== Phase 1.2: Scaling Attack GPU Experiments (27 total) ==="
echo "  GPU_ID=${GPU_ID}"
echo "  PMR=${PMR} → K=1, defense=none"
echo "  max_samples=0 (unlimited), test_subset=0 (full test set)"
echo ""

TOTAL=0
FAILED=0

# --- 正式实验 γ=10 ---

echo "--- Task A: ResNet18 + CIFAR-10 (100 rounds, γ=10) ---"
for ALPHA in $ALPHAS; do
	for SEED in $SEEDS; do
		TOTAL=$((TOTAL + 1))
		echo ""
		echo "[P1-SA-CIFAR10-g10] #${TOTAL}/27 alpha=${ALPHA} seed=${SEED} pmr=${PMR} γ=10"
		echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"
		if ! bash "$DIR/run_experiment.sh" \
			--model ResNet18 --dataset cifar10 \
			--attack model_replacement --defense none \
			--pmr "${PMR}" --alpha "${ALPHA}" --seed "${SEED}" \
			--rounds 100 --clients "${CLIENTS}" --epochs 1 --batch_size 64 \
			--max_samples 0 --test_subset 0 \
			--scale_gamma 10 \
			--gpu --gpu_id "${GPU_ID}" --runtime single-gpu-deterministic \
			--gpu_mapping mapping_single_gpu; then
			echo "  WARNING: experiment #${TOTAL} failed"
			FAILED=$((FAILED + 1))
		fi
		echo "  End: $(date '+%Y-%m-%d %H:%M:%S')"
	done
done

echo ""
echo "--- Task B: LeNet5 + MNIST (50 rounds, γ=10) ---"
for ALPHA in $ALPHAS; do
	for SEED in $SEEDS; do
		TOTAL=$((TOTAL + 1))
		echo ""
		echo "[P1-SA-MNIST-g10] #${TOTAL}/27 alpha=${ALPHA} seed=${SEED} pmr=${PMR} γ=10"
		echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"
		if ! bash "$DIR/run_experiment.sh" \
			--model LeNet5 --dataset mnist \
			--attack model_replacement --defense none \
			--pmr "${PMR}" --alpha "${ALPHA}" --seed "${SEED}" \
			--rounds 50 --clients "${CLIENTS}" --epochs 1 --batch_size 64 \
			--max_samples 0 --test_subset 0 \
			--scale_gamma 10 \
			--gpu --gpu_id "${GPU_ID}" --runtime single-gpu-deterministic \
			--gpu_mapping mapping_single_gpu; then
			echo "  WARNING: experiment #${TOTAL} failed"
			FAILED=$((FAILED + 1))
		fi
		echo "  End: $(date '+%Y-%m-%d %H:%M:%S')"
	done
done

# --- 隔离 γ=10 α=0.5 结果，防止 γ=1 控制组追加写入同文件 ---
# MetricsCollector 文件名不含 γ，γ=10 和 γ=1 在 α=0.5 时文件名相同。
# 在运行 γ=1 前，将冲突的 3 个文件移到子目录。

RESULTS_DIR="${DIR}/../results"
GAMMA10_ARCHIVE="${RESULTS_DIR}/p1_sa_gamma10_a0.5"
mkdir -p "${GAMMA10_ARCHIVE}"

echo ""
echo "--- Archiving γ=10 α=0.5 results to avoid γ=1 filename collision ---"
for SEED in $SEEDS; do
	SRC="${RESULTS_DIR}/metrics_ResNet18_cifar10_shieldfl_atkmodel_replacement_defnone_a0.5_pmr${PMR}_seed${SEED}.jsonl"
	if [[ -f "$SRC" ]]; then
		mv "$SRC" "${GAMMA10_ARCHIVE}/"
		echo "  Moved: $(basename "$SRC") → p1_sa_gamma10_a0.5/"
	fi
done

# --- 控制实验 γ=1 ---

echo ""
echo "--- Task C: ResNet18 + CIFAR-10 (100 rounds, γ=1, alpha=0.5) [控制组] ---"
for SEED in $SEEDS; do
	TOTAL=$((TOTAL + 1))
	echo ""
	echo "[P1-SA-CIFAR10-g1-CTRL] #${TOTAL}/27 alpha=0.5 seed=${SEED} pmr=${PMR} γ=1"
	echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"
	if ! bash "$DIR/run_experiment.sh" \
		--model ResNet18 --dataset cifar10 \
		--attack model_replacement --defense none \
		--pmr "${PMR}" --alpha "0.5" --seed "${SEED}" \
		--rounds 100 --clients "${CLIENTS}" --epochs 1 --batch_size 64 \
		--max_samples 0 --test_subset 0 \
		--scale_gamma 1 \
		--gpu --gpu_id "${GPU_ID}" --runtime single-gpu-deterministic \
		--gpu_mapping mapping_single_gpu; then
		echo "  WARNING: experiment #${TOTAL} failed"
		FAILED=$((FAILED + 1))
	fi
	echo "  End: $(date '+%Y-%m-%d %H:%M:%S')"
done

echo ""
echo "=========================================="
echo "=== Phase 1.2 SA done: ${TOTAL} experiments, ${FAILED} failed ==="
echo "=========================================="
