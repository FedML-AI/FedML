#!/usr/bin/env bash

GPU=$1

CLIENT_NUM=$2

WORKER_NUM=$3

BATCH_SIZE=$4

DATASET=$5

DATA_PATH=$6

MODEL=$7

DISTRIBUTION=$8

ROUND=$9

EPOCH=$10

LR=$11

WD=$12

GMF=$13

MU=$14

MOMENTUM=$15

DAMPENING=$16

WD=$17

NESTEROV=$18

CI=$19

python3 ./main_fednova.py \
--gpu $GPU \
--dataset $DATASET \
--data_dir $DATA_PATH \
--model $MODEL \
--partition_method $DISTRIBUTION  \
--client_num_in_total $CLIENT_NUM \
--client_num_per_round $WORKER_NUM \
--comm_round $ROUND \
--epochs $EPOCH \
--batch_size $BATCH_SIZE \
--lr $LR \
--wd $WD \
--gmf $GMF \
--mu $MU \
--momentum $MOMENTUM \
--dampening $DAMPENING \
--nesterov $NESTEROV \
--ci $CI
