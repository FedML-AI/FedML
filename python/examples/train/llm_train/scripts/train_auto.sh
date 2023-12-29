#!/bin/bash
set -e

BASE_DIR="$(dirname "$0")"
BASE_DIR="$(realpath "${BASE_DIR}/../")"
cd "${BASE_DIR}"

MASTER_ADDR="${1:-"localhost"}"
MASTER_PORT="${2:-12355}"
NUM_NODES="${3:-1}"
NUM_GPU="$(python3 -c "import torch; print(torch.cuda.device_count())")"

EXTRA_ARGS=()

# infer runner based on GPU count
if [[ "${NUM_GPU}" -gt 1 ]]; then
  echo "Detected ${NUM_GPU} GPUs. Use PyTorch distributed runner."

  if python3 -c "import deepspeed" &>/dev/null; then
    echo "Found DeepSpeed. Use DeepSpeed for training."

    EXTRA_ARGS+=(
      --deepspeed "configs/deepspeed/ds_z3_bf16_config.json"
    )
  fi

  CMD=(
    torchrun
    --nnodes="${NUM_NODES}"
    --nproc_per_node="${NUM_GPU}"
    --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}"
  )
else
  echo "Use default python runner."

  CMD=(
    python3
  )
fi

"${CMD[@]}" \
  run_train.py \
  "${EXTRA_ARGS[@]}" \
  --ddp_find_unused_parameters "False" \
  --model_name_or_path "EleutherAI/pythia-70m" \
  --dataset_name "FedML/databricks-dolly-15k-niid" \
  --seed 1234 \
  --fp16 "False" \
  --bf16 "False" \
  --peft_type "lora" \
  --gradient_checkpointing "True" \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --optim "adamw_torch" \
  --lr_scheduler_type "cosine" \
  --learning_rate "3e-4" \
  --warmup_steps 50 \
  --num_train_epochs 3 \
  --output_dir ".logs/dolly_pythia-70m" \
  --logging_steps 20 \
  --eval_steps 200 \
  --save_steps 200 \
  --save_total_limit 10 \
  --logging_strategy "steps" \
  --evaluation_strategy "steps" \
  --save_strategy "steps" \
  --eval_accumulation_steps 4 \
  --do_train "True" \
  --do_eval "True" \
  --do_predict "True" \
  --remove_long_seq "True" \
  "${@:4}" # skip first 3 arguments
