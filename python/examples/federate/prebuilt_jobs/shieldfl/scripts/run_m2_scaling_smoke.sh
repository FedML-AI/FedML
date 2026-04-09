#!/usr/bin/env bash
# M2 Scaling Attack Smoke Test (5 轮)
#
# 使用方法：
#   cd python/examples/federate/prebuilt_jobs/shieldfl
#   bash scripts/run_m2_scaling_smoke.sh
#
# 自动写入：
#   attack_training_rounds=[3, 4]
#   byzantine_client_num=3
#   scale_gamma=10
#   backdoor_per_batch=20
#
# 参考: Scaling_实施定稿.md §7.1

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

bash "${SCRIPT_DIR}/run_experiment.sh" \
  --model ResNet18 --dataset cifar10 \
  --attack model_replacement --defense none --aggregator fedavg \
  --pmr 0.3 --alpha 0.5 --seed 0 \
  --rounds 5 --clients 10 --epochs 1 --batch_size 64 \
  --gpu --gpu_mapping mapping_single_gpu --runtime single-gpu-deterministic
