#!/usr/bin/env bash



CLIENT_NUM=8
WORKER_NUM=8
MODEL=resnet56
DISTRIBUTION=homo
ROUND=3
EPOCH=1
BATCH_SIZE=64
LR=0.001
DATASET=cifar10
DATA_DIR="./../../../data/cifar10"
CLIENT_OPTIMIZER=adam
GRPC_CONFIG_PATH="./communication_benchmark/grpc/grpc_ipconfig.csv"
CI=0
BACKEND=GRPC



PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM


unset http_proxy
unset https_proxy


date=$(date +%s)
logfile="./grpc.$date.log"
echo "Using _transport cuda_gdr" >> ./$logfile
#
#(cd ../.. && mpirun -np $PROCESS_NUM -hostfile ./communication_benchmark/grpc/mpi_host_file python3 ./main_fedavg_rpc_mpi.py \
#  --gpu_mapping_file "./communication_benchmark/grpc/gpu_mapping.yaml" \
#  --gpu_mapping_key "mapping_FedMLـgRPC" \
#  --model $MODEL \
#  --dataset $DATASET \
#  --data_dir $DATA_DIR \
#  --partition_method $DISTRIBUTION  \
#  --client_num_in_total $CLIENT_NUM \
#  --client_num_per_round $WORKER_NUM \
#  --comm_round $ROUND \
#  --epochs $EPOCH \
#  --client_optimizer $CLIENT_OPTIMIZER \
#  --batch_size $BATCH_SIZE \
#  --lr $LR \
#  --backend $BACKEND \
#  --grpc_ipconfig_path $GRPC_CONFIG_PATH \
#  --ci $CI
#) >> $logfile 2>&1

mpirun -np $PROCESS_NUM -hostfile mpi_host_file python3 ./main_fedavg_rpc_mpi.py \
  --gpu_mapping_file "gpu_mapping.yaml" \
  --gpu_mapping_key "mapping_FedML_gRPC" \
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









