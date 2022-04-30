#!/usr/bin/env bash

CLIENT_NUM=$1
WORKER_NUM=$2
SERVER_NUM=$3
GPU_NUM_PER_SERVER=$4
MODEL=$5
DISTRIBUTION=$6
ROUND=$7
EPOCH=$8
BATCH_SIZE=$9
LR=${10}
DATASET=${11}
DATA_DIR=${12}
CLIENT_OPTIMIZER=${13}
CI=${14}
PYTHON=${15}
GPU=${16}
GPU_UTIL=${17} 
NPROC_PER_NODE=${18}


echo $BATCH_SIZE
echo $LR


$PYTHON -m torch.distributed.launch \
    --nproc_per_node=$NPROC_PER_NODE \
    ./main.py \
    --gpu_server_num $SERVER_NUM \
    --gpu_num_per_server $GPU_NUM_PER_SERVER \
    --model $MODEL \
    --dataset $DATASET \
    --data_dir $DATA_DIR \
    --partition_method $DISTRIBUTION  \
    --client_num_in_total $CLIENT_NUM \
    --client_num_per_round $WORKER_NUM \
    --comm_round $ROUND \
    --epochs $EPOCH \
    --client_optimizer $CLIENT_OPTIMIZER \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --ci $CI \
    --gpu $GPU \
    --gpu_util $GPU_UTIL \
    --data_parallel 1 \
