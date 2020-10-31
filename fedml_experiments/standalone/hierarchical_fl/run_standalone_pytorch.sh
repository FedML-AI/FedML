#!/usr/bin/env bash

GPU=$1

CLIENT_NUM=$2

WORKER_NUM=$3

BATCH_SIZE=$4

DATASET=$5

DATA_PATH=$6

MODEL=$7

DISTRIBUTION=$8

LR=$9

OPT=$10

GROUP_METHOD=$11

GROUP_NUM=$12

GLOBAL_COMM_ROUND=$13

GROUP_COMM_ROUND=$14

EPOCH=$15

python3 ./main.py \
--gpu $GPU \
--client_num_in_total $CLIENT_NUM \
--client_num_per_round $WORKER_NUM \
--batch_size $BATCH_SIZE \
--dataset $DATASET \
--data_dir $DATA_PATH \
--model $MODEL \
--partition_method $DISTRIBUTION  \
--lr $LR \
--client_optimizer $OPT \
--group_method $GROUP_METHOD \
--group_num $GROUP_NUM \
--global_comm_round $GLOBAL_COMM_ROUND \
--group_comm_round $GROUP_COMM_ROUND \
--epochs $EPOCH