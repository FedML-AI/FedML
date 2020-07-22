#!/usr/bin/env bash

GPU=$1
# homo; hetero
DISTRIBUTION=$2

ROUND=$3

EPOCH=$4

ARCH=$5

python3 ./main.py \
--gpu $GPU \
--dataset cifar10 \
--partition $DISTRIBUTION  \
--stage "train" \
--client_number 16 \
--comm_round $ROUND \
--epochs $EPOCH \
--batch-size 64 \
--arch $ARCH


# script for debugging
# sh run_fednas_train.sh 0 hetero 2 2 FedNAS_V1 > log.txt 2>&1 &