#!/usr/bin/env bash

GPU=$1

# homo; hetero
DISTRIBUTION=$2

ROUND=$3

EPOCH=$4

python3 ./main.py \
--gpu $GPU \
--dataset cifar10 \
--partition $DISTRIBUTION  \
--client_number 16 \
--comm_round $ROUND \
--epochs $EPOCH \
--batch-size 64