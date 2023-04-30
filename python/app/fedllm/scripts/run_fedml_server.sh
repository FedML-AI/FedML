#!/usr/bin/env bash
set -e

BASE_DIR="$(dirname "$0")"
BASE_DIR="$(realpath "${BASE_DIR}/../")"
cd "${BASE_DIR}"

export WANDB_MODE=disabled # remove this line if you want to use wandb
export CUDA_VISIBLE_DEVICES="0"

# FedML related
RANK=0
RUN_ID="$1"

# FedML config
CONFIG_PATH="fedml_config/fedml_server_config.yaml"

python main_federated_llm.py \
  --cf "${CONFIG_PATH}" \
  --rank "${RANK}" \
  --role server \
  --run_id "${RUN_ID}"
