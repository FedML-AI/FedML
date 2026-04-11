#!/usr/bin/env bash
# M1：基线可信 CPU smoke test
# 两条任务线：ResNet18 + CIFAR10 / LeNet5 + MNIST
# 轻量配置：3 轮，3 客户端
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

run() {
  bash "$DIR/run_experiment.sh" \
    --model "$1" --dataset "$2" \
    --attack none --defense none --aggregator fedavg \
    --pmr 0.0 --alpha 0.5 --seed "$3" \
    --rounds 3 --clients 3 --epochs 1 --batch_size 32 \
    --max_samples 300
}

echo "=== M1 Baseline smoke test ==="
for SEED in 0 1; do
  echo "--- ResNet18 + CIFAR10 | seed=${SEED} ---"
  run "ResNet18" "cifar10" "$SEED"

  echo "--- LeNet5 + MNIST | seed=${SEED} ---"
  run "LeNet5" "mnist" "$SEED"
done

echo "=== M1 done ==="
