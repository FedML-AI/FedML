#!/usr/bin/env bash

CLIENT_NUM=$1
WORKER_NUM=$2
SERVER_NUM=$3
GPU_NUM_PER_SERVER=$4
MODEL=$5
BACKBONE=$6
BACKBONE_PRETRAINED=$7
OUTPUT_STRIDE=$8
CATEGORIES=$9
DISTRIBUTION=$10
ROUND=$11
EPOCH=$12
BATCH_SIZE=$13
SYNC_BN=$14
FREEZE_BN=$15
CLIENT_OPTIMIZER=$16
LR=$17
LR_SCHEDULER=$18
DATASET=$18
DATA_DIR=$19
CI=$20

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
  --categories $CATEGORIES \
  --dataset $DATASET \
  --data_dir $DATA_DIR \
  --partition_method $DISTRIBUTION  \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $WORKER_NUM \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --batch_size $BATCH_SIZE \
  --sync_bn $SYNC_BN
  --freeze_bn $FREEZE_BN
  --client_optimizer $CLIENT_OPTIMIZER \
  --lr $LR \
  --lr_scheduler $LR_SCHEDULER \
  --ci $CI