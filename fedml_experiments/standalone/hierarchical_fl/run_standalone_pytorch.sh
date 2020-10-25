#!/usr/bin/env bash

GPU=$1

CLIENT_NUM=$2

WORKER_NUM=$3

BATCH_SIZE=$4

DATASET=$5

DATA_PATH=$6

MODEL=$7

DISTRIBUTION=$8

GLOBAL_COMM_ROUND=$9

GROUP_COMM_ROUND=$10

EPOCH=$11

LR=$12

OPT=$13

GROUP_METHOD=$14

GROUP_NUM=$15

CI=$16

python3 ./main.py \
--gpu $GPU \
--dataset $DATASET \
--data_dir $DATA_PATH \
--model $MODEL \
--partition_method $DISTRIBUTION  \
--client_num_in_total $CLIENT_NUM \
--client_num_per_round $WORKER_NUM \
--global_comm_round $GLOBAL_COMM_ROUND \
--group_comm_round $GROUP_COMM_ROUND \
--epochs $EPOCH \
--batch_size $BATCH_SIZE \
--client_optimizer $OPT \
--lr $LR \
--group_method $GROUP_METHOD \
--group_num $GROUP_NUM \
--ci $CI
