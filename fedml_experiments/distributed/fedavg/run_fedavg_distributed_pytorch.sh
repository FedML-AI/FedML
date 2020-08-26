#!/usr/bin/env bash

WORKER_NUM=$1
SERVER_NUM=$2
GPU_NUM_PER_SERVER=$3
MODEL=$4
DISTRIBUTION=$5
ROUND=$6
EPOCH=$7
BATCH_SIZE=$8
LR=$9
DATASET=$10
DATA_DIR=$11

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 ./main_fedavg.py \
  --gpu_server_num $SERVER_NUM \
  --gpu_num_per_server $GPU_NUM_PER_SERVER \
  --model $MODEL \
  --dataset $DATASET \
  --data_dir $DATA_DIR \
  --partition_method $DISTRIBUTION  \
  --client_number $WORKER_NUM \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --batch_size $BATCH_SIZE \
  --lr $LR