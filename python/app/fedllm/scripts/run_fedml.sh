#!/usr/bin/env bash
set -e

BASE_DIR="$(dirname "$0")"
BASE_DIR="$(realpath "${BASE_DIR}/../")"
cd "${BASE_DIR}"

if [[ -z "${WANDB_MODE}" ]]; then
  export WANDB_MODE=disabled # remove this line if you want to use wandb
fi

NUM_GPU="$(python3 -c "import torch; print(torch.cuda.device_count())")"

MASTER_ADDR="${1:-"localhost"}"
MASTER_PORT="${2:-29500}"
NUM_NODES="${3:-1}"

echo "CUDA_VISIBLE_DEVICES=\"${CUDA_VISIBLE_DEVICES}\""
echo "${MASTER_ADDR}:${MASTER_PORT},${NUM_GPU},${NUM_NODES}"

if [ "${NUM_GPU}" -gt 0 ]; then
  CMD=(
    deepspeed
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
  )

  if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
    # when `CUDA_VISIBLE_DEVICES` is not specified, use all GPUs by setting `--num_nodes`
    CMD+=(
      --num_nodes="${NUM_NODES}"
      --num_gpus="${NUM_GPU}"
    )
  else
    # see https://github.com/microsoft/DeepSpeed/issues/662
    # use `--include` to select GPUs and unset `CUDA_VISIBLE_DEVICES`
    CMD+=(
      --include "${MASTER_ADDR}:${CUDA_VISIBLE_DEVICES}"
    )
    unset CUDA_VISIBLE_DEVICES
  fi
else
  CMD=(
    python3
  )
fi

"${CMD[@]}" "${@:4}"
