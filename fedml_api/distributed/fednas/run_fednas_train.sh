#!/usr/bin/env bash

GPU=$1
MODEL=$2
# homo; hetero
DISTRIBUTION=$3
ROUND=$4
EPOCH=$5
BATCH_SIZE=$6

mpirun -np 17 -hostfile ./mpi_host_file python3 ./main.py \
  --gpu $GPU \
  --stage "train" \
  --model $MODEL \
  --dataset cifar10 \
  --partition $DISTRIBUTION  \
  --client_number 16 \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --batch_size $BATCH_SIZE