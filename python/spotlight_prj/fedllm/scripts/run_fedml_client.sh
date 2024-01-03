#!/usr/bin/env bash
set -e

BASE_DIR="$(dirname "$0")"
BASE_DIR="$(realpath "${BASE_DIR}/../")"
cd "${BASE_DIR}"

if [[ -z "${WANDB_MODE}" ]]; then
  export WANDB_MODE=disabled # remove this line if you want to use wandb
fi

# FedML setting
RANK="$1"
RUN_ID="$2"

# GPU setting
TORCH_DISTRIBUTED_DEFAULT_PORT="${TORCH_DISTRIBUTED_DEFAULT_PORT:-29500}"

MASTER_ADDR="${3:-"localhost"}"
MASTER_PORT="${4:-$((TORCH_DISTRIBUTED_DEFAULT_PORT + RANK))}"
NUM_NODES="${5:-1}"
LAUNCHER="${6:-"auto"}"

# FedML config
CONFIG_PATH="${7:-"fedml_config/fedml_config.yaml"}"

python3 launch_fedllm.py \
  --cf "${CONFIG_PATH}" \
  --rank "${RANK}" \
  --role client \
  --run_id "${RUN_ID}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  --num_nodes "${NUM_NODES}" \
  --launcher "${LAUNCHER}"
