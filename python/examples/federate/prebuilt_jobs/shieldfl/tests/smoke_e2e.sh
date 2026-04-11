#!/usr/bin/env bash
# T7: 端到端短训练冒烟测试
# 在 GPU 上跑 3 轮 FedAvg (无防御) + 3 轮 Krum 防御，验证训练流程完整
# 用法: bash tests/smoke_e2e.sh --gpu_id 1
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

GPU_ID="${1:-1}"
if [[ "$1" == "--gpu_id" ]]; then GPU_ID="${2:-1}"; fi

echo "============================================================"
echo "T7: End-to-end smoke tests (GPU ${GPU_ID})"
echo "============================================================"

PASS=0
FAIL=0

run_test() {
	local name="$1"
	shift
	echo ""
	echo "--- ${name} ---"
	echo "CMD: $*"
	if "$@"; then
		echo "[PASS] ${name}"
		PASS=$((PASS + 1))
	else
		echo "[FAIL] ${name} (exit=$?)"
		FAIL=$((FAIL + 1))
	fi
}

# T7.1: FedAvg baseline (no attack, no defense), 3 rounds, 3 clients
run_test "T7.1 FedAvg-no-defense-3r" \
	bash scripts/run_experiment.sh \
	--model SimpleCNN --dataset cifar10 \
	--attack none --defense none \
	--pmr 0.0 --alpha 0.5 --seed 42 \
	--rounds 3 --clients 3 --epochs 1 --batch_size 64 \
	--max_samples 200 --test_subset 200 \
	--gpu --gpu_id "${GPU_ID}"

# T7.2: FedAvg + Krum defense, 3 rounds
run_test "T7.2 FedAvg-krum-3r" \
	bash scripts/run_experiment.sh \
	--model SimpleCNN --dataset cifar10 \
	--attack none --defense krum \
	--pmr 0.0 --alpha 0.5 --seed 42 \
	--rounds 3 --clients 5 --epochs 1 --batch_size 64 \
	--max_samples 200 --test_subset 200 \
	--gpu --gpu_id "${GPU_ID}" --gpu_mapping mapping_5clients

# T7.3: FedAvg + model_replacement attack + no defense (verify attack pipeline)
run_test "T7.3 FedAvg-model_replacement-no-defense-3r" \
	bash scripts/run_experiment.sh \
	--model SimpleCNN --dataset cifar10 \
	--attack model_replacement --defense none \
	--pmr 0.2 --alpha 0.5 --seed 42 \
	--rounds 3 --clients 5 --epochs 1 --batch_size 64 \
	--max_samples 200 --test_subset 200 \
	--gpu --gpu_id "${GPU_ID}" --gpu_mapping mapping_5clients

# T7.4: FedAvg + model_replacement attack + Krum defense
run_test "T7.4 FedAvg-model_replacement-krum-3r" \
	bash scripts/run_experiment.sh \
	--model SimpleCNN --dataset cifar10 \
	--attack model_replacement --defense krum \
	--pmr 0.2 --alpha 0.5 --seed 42 \
	--rounds 3 --clients 5 --epochs 1 --batch_size 64 \
	--max_samples 200 --test_subset 200 \
	--gpu --gpu_id "${GPU_ID}" --gpu_mapping mapping_5clients

echo ""
echo "============================================================"
echo "T7 Summary: ${PASS} passed, ${FAIL} failed"
echo "============================================================"

if [[ "$FAIL" -gt 0 ]]; then
	exit 1
fi
