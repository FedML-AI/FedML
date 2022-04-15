#!/usr/bin/env bash
set -x

# run as client
SILO_RANK=1
RUN_ID=169

GPU_MAPPING_FILE="./gpu_mapping.yaml"
GPU_MAPPING_KEY="mapping_FedML_cross_silo"
SILO_GPU_MAPPING_FILE="./silo_gpu_mapping_client.yaml"
ITERATION_PER_ROUND=80
SILO_NUM=1
NPROC_PER_NODE=2
NNODE=1
NODE_RANK=0
MODEL=resnet56
DISTRIBUTION=homo
ROUND=3
EPOCH=1
BATCH_SIZE=64
LR=0.001
DATASET=cifar10
DATA_DIR="./../../../data/cifar10"
CLIENT_OPTIMIZER=adam
BACKEND=MQTT_S3
TRPC_MASTER_CONFIG_PATH="./trpc_master_config.csv"
CI=0
PG_MASTER_ADDRESS="127.0.0.1"
PG_MASTER_PORT="29501"
S3_CONFIG_PATH="./s3_config.yaml"
MQTT_CONFIG_PATH="./mqtt_config.yaml"


 #--gpu_mapping_file $GPU_MAPPING_FILE \
  #--gpu_mapping_key $GPU_MAPPING_KEY \
  #--silo_gpu_mapping_file $SILO_GPU_MAPPING_FILE \
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
  --nnode $NNODE \
  --silo_node_rank $NODE_RANK \
  --nproc_per_node $NPROC_PER_NODE \
  --silo_rank $SILO_RANK \
  --pg_master_address $PG_MASTER_ADDRESS\
  --pg_master_port $PG_MASTER_PORT \
  --s3_config_path $S3_CONFIG_PATH \
  --mqtt_config_path $MQTT_CONFIG_PATH \
  --network_interface en0 \
  --run_id $RUN_ID \
  --client_ids [17]