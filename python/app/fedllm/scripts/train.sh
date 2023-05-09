#!/bin/bash
set -e

BASE_DIR="$(dirname "$0")"
BASE_DIR="$(realpath "${BASE_DIR}/../")"
cd "${BASE_DIR}"

bash scripts/setup.sh

export CUDA_VISIBLE_DEVICES="0"

DATASET_PATHS=(
  ".data/databricks-dolly-15k.jsonl"
  # add your datasets here
)

python3 train.py \
  --model_name "EleutherAI/pythia-6.9b" \
  --dataset_path "${DATASET_PATHS[@]}" \
  --test_dataset_size 200 \
  --seed 1234 \
  --fp16 "False" \
  --bf16 "False" \
  --use_lora \
  --gradient_checkpointing "True" \
  --per_device_train_batch_size 6 \
  --per_device_eval_batch_size 6 \
  --learning_rate "5e-6" \
  --warmup_steps 50 \
  --num_train_epochs 5 \
  --output_dir ".logs/dolly_pythia-6.9b" \
  --logging_steps 50 \
  --eval_steps 200 \
  --save_steps 200 \
  --save_total_limit 20 \
  --logging_strategy "steps" \
  --evaluation_strategy "steps" \
  --save_strategy "steps" \
  --eval_accumulation_steps 4 \
  --do_train "True" \
  "${@}"
