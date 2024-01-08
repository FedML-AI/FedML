#!/usr/bin/env bash
set -e

BASE_DIR="$(dirname "$0")"
BASE_DIR="$(realpath "${BASE_DIR}/../")"
cd "${BASE_DIR}"

if [[ -z "${WANDB_MODE}" ]]; then
  export WANDB_MODE=disabled # remove this line if you want to use wandb
fi

# FedML setting
RANK=0
RUN_ID="$1"

# GPU setting
TORCH_DISTRIBUTED_DEFAULT_PORT="${TORCH_DISTRIBUTED_DEFAULT_PORT:-29500}"

MASTER_ADDR="${2:-"localhost"}"
MASTER_PORT="${3:-$((TORCH_DISTRIBUTED_DEFAULT_PORT + RANK))}"
NUM_NODES="${4:-1}"
LAUNCHER="${5:-"auto"}"

# FedML config
CONFIG_PATH="${6:-"fedml_config/fedml_config.yaml"}"

python3 launch_fedllm.py \
  --cf "${CONFIG_PATH}" \
  --rank "${RANK}" \
  --role server \
  --run_id "${RUN_ID}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  --num_nodes "${NUM_NODES}" \
  --launcher "${LAUNCHER}"
