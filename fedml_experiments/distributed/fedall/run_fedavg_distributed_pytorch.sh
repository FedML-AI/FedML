#!/usr/bin/env bash

CLIENT_NUM=$1
WORKER_NUM=$2
SERVER_NUM=$3
GPU_NUM_PER_SERVER=$4
MODEL=$5
DISTRIBUTION=$6
ROUND=$7
EPOCH=$8
BATCH_SIZE=$9
LR=${10}
DATASET=${11}
DATA_DIR=${12}
CLIENT_OPTIMIZER=${13}
CI=${14}
GPU_UTIL_FILE=${15}
MPI_HOST_FILE=${16}
PYTHON=${17}

echo $GPU_UTIL_FILE



PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

# hostname > mpi_host_file

mpirun -np $PROCESS_NUM -hostfile ./$MPI_HOST_FILE $PYTHON ./main_fedavg.py \
  --gpu_server_num $SERVER_NUM \
  --gpu_num_per_server $GPU_NUM_PER_SERVER \
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
  --ci $CI \
  --gpu_util_file $GPU_UTIL_FILE
