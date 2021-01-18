#!/usr/bin/env bash

CLIENT_NUM=$1
WORKER_NUM=$2
SERVER_NUM=$3
GPU_NUM_PER_SERVER=$4
MODEL=$5
ROUND=$6
EPOCH=$7
BATCH_SIZE=$8
LR=${9}
DATASET=${10}
DATA_DIR=${11}
SERVER_OPTIMIZER=${12}
CI=${13}

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo process num : $PROCESS_NUM
hostname > mpi_host_file

mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 ./main_fedopt.py \
  --gpu_server_num $SERVER_NUM \
  --gpu_num_per_server $GPU_NUM_PER_SERVER \
  --model $MODEL \
  --dataset $DATASET \
  --data_dir $DATA_DIR \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $WORKER_NUM \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --server_optimizer $SERVER_OPTIMIZER \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --ci $CI
