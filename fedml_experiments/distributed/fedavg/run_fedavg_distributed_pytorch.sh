#!/usr/bin/env bash

SERVER_NUM=$1
GPU_NUM_PER_SERVER=$2
MODEL=$3
DISTRIBUTION=$4
ROUND=$5
EPOCH=$6
BATCH_SIZE=$7
LR=$8
DATASET=$9
DATA_DIR=$10


mpirun -np 11 -hostfile ./mpi_host_file python3 ./main_fedavg.py \
  --gpu_server_num $SERVER_NUM \
  --gpu_num_per_server $GPU_NUM_PER_SERVER \
  --model $MODEL \
  --dataset $DATASET \
  --data_dir $DATA_DIR \
  --partition_method $DISTRIBUTION  \
  --client_number 10 \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --batch_size $BATCH_SIZE \
  --lr $LR