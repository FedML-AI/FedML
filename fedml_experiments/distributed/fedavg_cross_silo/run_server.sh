#!/usr/bin/env bash
set -x

# compute topology related arguments (silo global)
SILO_RANK=0
SILO_NUM=3
RUN_ID=168

BACKEND=MQTT_S3

# BACKEND: MQTT_S3
S3_CONFIG_PATH="./s3_config.yaml" # optional; only used when PyTorch RPC is enabled
MQTT_CONFIG_PATH="./mqtt_config.yaml" # optional; only used when PyTorch RPC is enabled

# BACKEND: TRPC
TRPC_MASTER_CONFIG_PATH="./trpc_master_config.csv"
GPU_MAPPING_FILE="./gpu_mapping.yaml" # optional; only used when PyTorch RPC is enabled
GPU_MAPPING_KEY="mapping_FedML_cross_silo" # optional; only used when PyTorch RPC is enabled

# compute topology related arguments (silo local)
SILO_GPU_MAPPING_FILE="./silo_gpu_mapping_server.yaml"
NPROC_PER_NODE=1
# TODO: have multiple nodes per silo
NNODE=1
# TODO: have multiple nodes per silo
NODE_RANK=0
PG_MASTER_ADDRESS="127.0.0.1"
PG_MASTER_PORT="29500"


#  training related arguments
MODEL=resnet56
DISTRIBUTION=homo
ROUND=3
EPOCH=1
BATCH_SIZE=64
LR=0.001
DATASET=cifar10
DATA_DIR="./../../../data/cifar10"
CLIENT_OPTIMIZER=adam

CI=0

#  --gpu_mapping_file $GPU_MAPPING_FILE \
#  --gpu_mapping_key $GPU_MAPPING_KEY \
#  --silo_gpu_mapping_file $SILO_GPU_MAPPING_FILE \

sh run_fedavg_cross_silo.sh \
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
  --ci $CI \
  --trpc_master_config_path $TRPC_MASTER_CONFIG_PATH \
  --silo_node_rank $NODE_RANK \
  --nproc_per_node $NPROC_PER_NODE \
  --silo_rank $SILO_RANK \
  --pg_master_address $PG_MASTER_ADDRESS\
  --pg_master_port $PG_MASTER_PORT\
  --s3_config_path $S3_CONFIG_PATH \
  --mqtt_config_path $MQTT_CONFIG_PATH \
  --network_interface lo \
  --run_id $RUN_ID \
  --client_ids [1]

  
# edge_id: [18, 21, 19, 20]
# mapping: [(18, 0), (21, 1), (19, 2), (20, 3)]
# client_indexes: [0, 1, 2, 3]