#!/usr/bin/env bash

GPU=$1

MODEL=$2

DISTRIBUTION=$3

ROUND=$4

EPOCH=$5

LR=$6


python3 ./main_fedavg.py \
--gpu $GPU \
--model $MODEL \
--dataset cifar10 \
--partition $DISTRIBUTION  \
--client_number 16 \
--comm_round $ROUND \
--epochs $EPOCH \
--batch-size 64 \
--lr $LR
