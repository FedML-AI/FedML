#!/usr/bin/env bash

CLIENT_NUM=$1
WORKER_NUM=$2
SERVER_NUM=$3
GPU_NUM_PER_SERVER=$4
MODEL=$5
BACKBONE=$6
BACKBONE_PRETRAINED=$7
OUTPUT_STRIDE=$8
DISTRIBUTION=$9
ROUND=$10
EPOCH=$11
BATCH_SIZE=$12
CLIENT_OPTIMIZER=$13
LR=$14
DATASET=$15
DATA_DIR=$16
CI=$17

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 ./main_fedseg.py \
  --gpu_server_num $SERVER_NUM \
  --gpu_num_per_server $GPU_NUM_PER_SERVER \
  --model $MODEL \
  --backbone $BACKBONE \
  --backbone_pretrained $BACKBONE_PRETRAINED \
  --outstride $OUTSTRIDE \
  --dataset $DATASET \
  --data_dir $DATA_DIR \
  --partition_method $DISTRIBUTION  \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $WORKER_NUM \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --batch_size $BATCH_SIZE \
  --client_optimizer $CLIENT_OPTIMIZER \
  --lr $LR \
  --ci $CI