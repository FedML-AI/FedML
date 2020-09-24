#!/usr/bin/env bash

WORKER_NUM=$1
GPU_NUM_PER_SERVER=$2
ROUND=$3
EPOCH=$4
LR=$5
MTL=$6
DISTRIBUTION=$7

hostname > mpi_host_file

mpirun -np $WORKER_NUM -hostfile ./mpi_host_file python3 ./main_mtl.py \
  --client_num_per_round $WORKER_NUM \
  --client_num_in_total $WORKER_NUM \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --is_mtl $MTL \
  --gpu_num_per_server $GPU_NUM_PER_SERVER \
  --lr $LR \
  --partition_method $DISTRIBUTION