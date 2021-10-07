#!/usr/bin/env bash



CLIENT_NUM=1
WORKER_NUM=1
MODEL=cnn
DISTRIBUTION=hetero
ROUND=1
EPOCH=1
BATCH_SIZE=20
LR=0.1
DATASET=femnist
DATA_DIR="./../../../data/FederatedEMNIST/datasets"
CLIENT_OPTIMIZER=sgd
GRPC_CONFIG_PATH="./communication_benchmark/grpc/grpc_ipconfig.csv"
CI=0
BACKEND=GRPC



PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM


unset http_proxy
unset https_proxy

(cd ../.. && mpirun -np $PROCESS_NUM -hostfile ./communication_benchmark/grpc/mpi_host_file python3 ./main_fedavg_rpc_mpi.py \
  --gpu_mapping_file "./communication_benchmark/grpc/gpu_mapping.yaml" \
  --gpu_mapping_key "mapping_FedMLـgRPC" \
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
  --grpc_ipconfig_path $GRPC_CONFIG_PATH \
  --ci $CI
)









