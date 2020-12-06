#!/usr/bin/env bash

CLIENT_NUM=$1
WORKER_NUM=$2
SERVER_NUM=$3
GPU_NUM_PER_SERVER=$4
MODEL=$5
DISTRIBUTION=$6
EPOCH=$7
BATCH_SIZE=$8
LR=${9}
DATASET=${10}
DATA_DIR=${11}
CLIENT_OPTIMIZER=${12}
CI=${13}
GPU=${14}

echo $BATCH_SIZE
echo $LR


python ./main.py \
  --gpu_server_num $SERVER_NUM \
  --gpu_num_per_server $GPU_NUM_PER_SERVER \
  --model $MODEL \
  --dataset $DATASET \
  --data_dir $DATA_DIR \
  --partition_method $DISTRIBUTION  \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $WORKER_NUM \
  --epochs $EPOCH \
  --client_optimizer $CLIENT_OPTIMIZER \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --ci $CI \
  --gpu $GPU















