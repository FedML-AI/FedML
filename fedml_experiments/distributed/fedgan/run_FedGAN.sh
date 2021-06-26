#!/usr/bin/env bash

CLIENT_NUM=$1
WORKER_NUM=$2
MODEL=$3
ROUND=$4
EPOCH=$5
BATCH_SIZE=$6
LR=$7

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 ./main_fedgan.py \
 --model $MODEL \
 --client_num_in_total $CLIENT_NUM \
 --client_num_per_round $WORKER_NUM \
 --comm_round $ROUND \
 --epochs $EPOCH \
 --batch_size $BATCH_SIZE \
 --lr $LR


