#!/usr/bin/env bash

GPU=$1

WORKER_NUM=$2

BATCH_SIZE=$3

DATASET=$4

DATA_PATH=$5

MODEL=$6

DISTRIBUTION=$7

ROUND=$8

EPOCH=$9

LR=$10

python3 ./main_fedavg.py \
--gpu $GPU \
--dataset $DATASET \
--data_dir $DATA_PATH \
--model $MODEL \
--partition_method $DISTRIBUTION  \
--client_num_per_round $WORKER_NUM \
--comm_round $ROUND \
--epochs $EPOCH \
--batch_size $BATCH_SIZE \
--lr $LR
