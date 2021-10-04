#!/usr/bin/env bash

CLIENT_NUM=10
WORKER_NUM=10
MODEL=resnet56
DISTRIBUTION=homo
ROUND=1
EPOCH=20
BATCH_SIZE=64
LR=0.001
DATASET=cifar100
DATA_DIR="./../../../data/cifar100"
CLIENT_OPTIMIZER=adam
BACKEND=TRPC
CI=0

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM


mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 ./main_fedavg.py \
  --gpu_mapping_file "gpu_mapping.yaml" \
  --gpu_mapping_key "mapping_default" \
  --model $MODEL \
  --dataset $DATASET \
  --data_dir $DATA_DIR \
  --partition_method $DISTRIBUTION  \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $WORKER_NUM \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --client_optimizer $CLIENT_OPTIMIZER \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --backend $BACKEND \
  --ci $CI
