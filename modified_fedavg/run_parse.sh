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

OPT=$11

CI=$12

python3 ./parse.py \
--gpu $GPU \
--dataset $DATASET \
--data_dir $DATA_PATH \
--model $MODEL \
--partition_method $DISTRIBUTION  \
--client_num_per_round $WORKER_NUM \
--comm_round $ROUND \
--epochs $EPOCH \
--batch_size $BATCH_SIZE \
--client_optimizer $OPT \
--lr $LR \
--ci $CI
