#!/usr/bin/env bash

SERVER_NUM=$1
GPU_NUM_PER_SERVER=$2
MODEL=$3
# homo; hetero
DISTRIBUTION=$4
ROUND=$5
EPOCH=$6
BATCH_SIZE=$7

hostname > mpi_host_file

mpirun -np 9 -hostfile ./mpi_host_file python3 ./main_fednas.py \
  --gpu_server_num $SERVER_NUM \
  --gpu_num_per_server $GPU_NUM_PER_SERVER \
  --model $MODEL \
  --stage "search" \
  --dataset cifar10 \
  --partition_method $DISTRIBUTION  \
  --client_number 8 \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --batch_size $BATCH_SIZE