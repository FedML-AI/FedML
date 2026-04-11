#!/usr/bin/env bash
# M3：防御对照 CPU smoke test
# VeriFL (aggregator) vs FedML 内置防御 × Byzantine 攻击
# 轻量配置：3 轮，5 客户端
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

run_baseline_defense() {
  bash "$DIR/run_experiment.sh" \
    --model "$1" --dataset "$2" \
    --attack byzantine --defense "$3" --aggregator fedavg \
    --pmr 0.2 --alpha 0.5 --seed "$4" \
    --rounds 3 --clients 5 --epochs 1 --batch_size 32 \
    --max_samples 300
}

run_verifl() {
  bash "$DIR/run_experiment.sh" \
    --model "$1" --dataset "$2" \
    --attack byzantine --defense none --aggregator verifl \
    --pmr 0.2 --alpha 0.5 --seed "$3" \
    --rounds 3 --clients 5 --epochs 1 --batch_size 32 \
    --max_samples 300
}

echo "=== M3 Defense smoke test ==="
SEED=0

echo "--- ResNet18 + CIFAR10 | VeriFL + Byzantine ---"
run_verifl "ResNet18" "cifar10" "$SEED"

for DEFENSE in krum trimmed_mean rfa; do
  echo "--- ResNet18 + CIFAR10 | defense=${DEFENSE} | seed=${SEED} ---"
  run_baseline_defense "ResNet18" "cifar10" "$DEFENSE" "$SEED"
done

echo "=== M3 done ==="
