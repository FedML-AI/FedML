#!/usr/bin/env bash
# M2：攻击生效 CPU smoke test
# 三类攻击 × FedAvg × 两条任务线
# 轻量配置：3 轮，5 客户端（PMR=20% → 1 恶意）
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

run() {
  bash "$DIR/run_experiment.sh" \
    --model "$1" --dataset "$2" \
    --attack "$3" --defense none --aggregator fedavg \
    --pmr 0.2 --alpha 0.5 --seed "$4" \
    --rounds 3 --clients 5 --epochs 1 --batch_size 32 \
    --max_samples 300
}

echo "=== M2 Attack smoke test ==="
for SEED in 0; do
  for ATTACK in byzantine label_flipping model_replacement; do
    echo "--- ResNet18 + CIFAR10 | attack=${ATTACK} | seed=${SEED} ---"
    run "ResNet18" "cifar10" "$ATTACK" "$SEED"

    echo "--- LeNet5 + MNIST | attack=${ATTACK} | seed=${SEED} ---"
    run "LeNet5" "mnist" "$ATTACK" "$SEED"
  done
done

echo "=== M2 done ==="
