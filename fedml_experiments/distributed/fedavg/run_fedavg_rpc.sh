#!/bin/bash
set -x

# usage of this script:

# the server (FL_WORKER_INDEX = 0)
# sh run_fedavg_cross_zone.sh 0

# client 0 (FL_WORKER_INDEX = 1)
# sh run_fedavg_cross_zone.sh 1

# client 1 (FL_WORKER_INDEX = 2)
# sh run_fedavg_cross_zone.sh 2

# infrastructure related
FL_WORKER_INDEX=$1
CLIENT_NUM=2
CLIENT_NUM_PER_ROUND=2
GPU_MAPPING="mapping_FedML_gRPC"

# dataset related
DATASET=mnist
DATA_DIR="./../../../data/MNIST"
DISTRIBUTION=hetero

# model and training related
MODEL=lr
ROUND=50
EPOCH=2
BATCH_SIZE=32
LR=0.01
CLIENT_OPTIMIZER=adam

hostname > mpi_host_file

unset http_proxy
unset https_proxy

python3 ./main_fedavg_rpc.py \
  --gpu_mapping_file "gpu_mapping.yaml" \
  --gpu_mapping_key $GPU_MAPPING \
  --model $MODEL \
  --dataset $DATASET \
  --data_dir $DATA_DIR \
  --partition_method $DISTRIBUTION  \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $CLIENT_NUM_PER_ROUND \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --client_optimizer $CLIENT_OPTIMIZER \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --ci 0 \
  --backend "GRPC" \
  --grpc_ipconfig_path "grpc_ipconfig.csv" \
  --fl_worker_index $FL_WORKER_INDEX
