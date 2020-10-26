#!/usr/bin/env bash

GPU=$1

DATASET=$2

# homo; hetero
DISTRIBUTION=$3

ROUND=$4

EPOCH_CLIENT=$5

EPOCH_SERVER=$6

OPTM=$7

LR=$8

TRAIN_OR_NOT=$9

DISTILL_ON_SERVER=$10

CLIENT_MODEL=$11

NAME=$12

DATA_DIR=$13

BATCH_SIZE=$14

hostname > mpi_host_file

mpirun -np 9 -hostfile ./mpi_host_file python3 ./main_fedgkt.py \
--gpu $GPU \
--dataset $DATASET \
--data_dir $DATA_DIR \
--partition_method $DISTRIBUTION  \
--client_number 8 \
--client_model $CLIENT_MODEL \
--comm_round $ROUND \
--epochs_client $EPOCH_CLIENT \
--epochs_server $EPOCH_SERVER \
--batch_size $BATCH_SIZE \
--optimizer $OPTM \
--lr $LR \
--weight_init_model resnet32 \
--whether_training_on_client $TRAIN_OR_NOT \
--whether_distill_on_the_server $DISTILL_ON_SERVER \
--running_name $NAME \
--multi_gpu_server

