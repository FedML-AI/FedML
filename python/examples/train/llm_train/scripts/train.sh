#!/bin/bash
set -e

BASE_DIR="$(dirname "$0")"
BASE_DIR="$(realpath "${BASE_DIR}/../")"
cd "${BASE_DIR}"

NUM_GPU="$(python3 -c "import torch; print(torch.cuda.device_count())")"

# see https://stackoverflow.com/a/13864829
if [[ -z "${CUDA_VISIBLE_DEVICES+x}" && "${NUM_GPU}" -gt 1 ]]; then
  echo "Detected ${NUM_GPU} > 1 GPUs; will use the first GPU."

  export CUDA_VISIBLE_DEVICES="0"
fi

python3 run_train.py \
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
  "${@}"
