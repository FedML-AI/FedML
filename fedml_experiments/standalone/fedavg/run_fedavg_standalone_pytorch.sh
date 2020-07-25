#!/usr/bin/env bash

GPU=$1

DATASET=$2

DATA_PATH=$3

MODEL=$4

DISTRIBUTION=$5

ROUND=$6

EPOCH=$7

LR=$8

python3 ./main_fedavg.py \
--gpu $GPU \
--dataset $DATASET \
--data_dir $DATA_PATH \
--model $MODEL \
--partition_method $DISTRIBUTION  \
--client_number 10 \
--comm_round $ROUND \
--epochs $EPOCH \
--batch-size 64 \
--lr $LR
