#!/usr/bin/env bash
# M2 LF 冒烟测试：5 轮短实验，验证 AC-7 ~ AC-9
# 运行方式: bash scripts/run_m2_lf_smoke.sh [--gpu_id 0]
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

GPU_ID="${1:-0}"
if [[ "$1" == "--gpu_id" ]]; then
	GPU_ID="$2"
fi

echo "=== LF Smoke Test (5 rounds, CIFAR-10, alpha=0.5, seed=0, PMR=0.3) ==="
echo "  GPU_ID=${GPU_ID}"

bash "$DIR/run_experiment.sh" \
	--model ResNet18 --dataset cifar10 \
	--attack label_flipping --defense none --aggregator fedavg \
	--pmr 0.3 --alpha 0.5 --seed 0 \
	--rounds 5 --clients 10 --epochs 1 --batch_size 64 \
	--max_samples 0 --test_subset 0 \
	--gpu --gpu_id "${GPU_ID}" --runtime single-gpu-deterministic \
	--gpu_mapping mapping_single_gpu

echo ""
echo "=== Smoke test complete. Check: ==="
echo "  1. Exit code 0 above"
echo "  2. JSONL metrics file in results/ with round 0~4"
echo "  3. Log lines containing [LabelFlippingAttack] for audit trail"
