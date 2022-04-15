#!/usr/bin/env bash
set -x


SILO_NUM=1
# TODO: have multiple nodes per silo
NNODE=1
# TODO: have multiple nodes per silo
NODE_RANK=0
MODEL=resnet56
DISTRIBUTION=homo
ROUND=2
EPOCH=1
BATCH_SIZE=64
LR=0.001
DATASET=cifar10
DATA_DIR="./../../../data/cifar10"
CLIENT_OPTIMIZER=adam
BACKEND=TRPC
TRPC_MASTER_CONFIG_PATH="./trpc_master_config.csv"
CI=0
PG_MASTER_ADDRESS="192.168.1.2"
PG_MASTER_PORT="29500"
NPROC_PER_NODE=1


sh run_fedavg_cross_silo.sh \
  --gpu_mapping_file "./gpu_mapping.yaml" \
  --gpu_mapping_key "mapping_FedML_cross_silo" \
  --silo_gpu_mapping_file "./silo_gpu_mapping_server.yaml" \
  --model $MODEL \
  --dataset $DATASET \
  --data_dir $DATA_DIR \
  --partition_method $DISTRIBUTION  \
  --client_silo_num_in_total $SILO_NUM \
  --silo_num_per_round $SILO_NUM \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --client_optimizer $CLIENT_OPTIMIZER \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --backend $BACKEND \
  --enable_cuda_rpc \
  --ci $CI \
  --trpc_master_config_path $TRPC_MASTER_CONFIG_PATH \
  --silo_node_rank $NODE_RANK \
  --nproc_per_node $NPROC_PER_NODE \
  --silo_rank 0 \
  --pg_master_address $PG_MASTER_ADDRESS\
  --pg_master_port $PG_MASTER_PORT\
  --network_interface eno2


  
