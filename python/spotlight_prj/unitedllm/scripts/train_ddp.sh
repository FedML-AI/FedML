#!/bin/bash
set -e

BASE_DIR="$(dirname "$0")"
BASE_DIR="$(realpath "${BASE_DIR}/../")"
cd "${BASE_DIR}"

MASTER_ADDR="${1:-"localhost"}"
MASTER_PORT="${2:-12355}"
NUM_NODES="${3:-1}"
NUM_GPU="$(python3 -c "import torch; print(torch.cuda.device_count())")"
RDZV_ID="666000"

#python3 \
#  -m torch.distributed.launch \
#  --nproc_per_node="${NUM_GPU}" \
#  --nnodes="${NUM_NODES}" \
#  --node_rank=0 \
#  --master_addr="${MASTER_ADDR}" \
#  --master_port="${MASTER_PORT} \
#  --use-env \
torchrun \
  --nnodes="${NUM_NODES}" \
  --nproc_per_node="${NUM_GPU}" \
  --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  --rdzv-id="${RDZV_ID}" \
  --rdzv-backend=c10d \
  run_train.py \
  --model_name_or_path "EleutherAI/pythia-70m" \
  --dataset_name "FedML/databricks-dolly-15k-niid" \
  --seed 1234 \
  --fp16 "False" \
  --bf16 "False" \
  --peft_type "loft" \
  --gradient_checkpointing "True" \
  --ddp_find_unused_parameters "False" \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --learning_rate "5e-6" \
  --warmup_steps 50 \
  --num_train_epochs 5 \
  --output_dir ".logs/dolly_pythia-70m" \
  --logging_steps 50 \
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
  --response_template "" \
  --truncate_long_seq "True" \
  --remove_long_seq "False" \
  "${@:4}" # skip first 3 arguments
