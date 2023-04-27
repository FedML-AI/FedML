#!/usr/bin/env bash
set -e

BASE_DIR="$(dirname "$0")"
BASE_DIR="$(realpath "${BASE_DIR}/../")"
cd "${BASE_DIR}"

# FedML related
RANK="$1"
RUN_ID="$2"

# GPU cluster related
NUM_NODES=1
NUM_GPU=4
TORCH_DISTRIBUTED_DEFAULT_PORT=29500
MASTER_PORT=`expr $TORCH_DISTRIBUTED_DEFAULT_PORT + $RANK`


deepspeed \
  --num_nodes="${NUM_NODES}" \
  --num_gpus="${NUM_GPU}" \
  --master_port="${MASTER_PORT}" \
  main_federated_llm.py \
  --cf fedml_config/fedml_config.yaml \
  --rank "${RANK}" \
  --role client \
  --run_id "${RUN_ID}"
