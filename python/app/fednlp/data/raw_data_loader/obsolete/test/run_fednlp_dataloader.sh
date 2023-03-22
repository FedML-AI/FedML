#!/usr/bin/env bash

DATASET=$1

DATA_PATH=$2

PARTITION_METHOD=$3

BATCH_SIZE=$4

CLIENT_NUM=$5

python3 ./test_dataloader.py \
--dataset $DATASET \
--data_dir $DATA_PATH \
--partition_method $PARTITION_METHOD \
--batch_size $BATCH_SIZE \
--client_num $CLIENT_NUM
