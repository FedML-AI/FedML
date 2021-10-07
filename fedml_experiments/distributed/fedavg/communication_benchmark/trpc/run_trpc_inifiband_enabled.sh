#!/usr/bin/env bash
set -x

# enable InfiniBand
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export TP_SOCKET_IFNAME=ib0
export NCCL_IB_HCA=ib0

# disable InfiniBand
# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=eno2
# export GLOO_SOCKET_IFNAME=eno2
# export TP_SOCKET_IFNAME=eno2

export NCCL_DEBUG=INFO
export NCCL_MIN_NRINGS=1
export NCCL_TREE_THRESHOLD=4294967296
export OMP_NUM_THREADS=8
export NCCL_NSOCKS_PERTHREAD=8
export NCCL_SOCKET_NTHREADS=8
export NCCL_BUFFSIZE=1048576


CLIENT_NUM=3
WORKER_NUM=3
MODEL=resnet56
DISTRIBUTION=homo
ROUND=10
EPOCH=2
BATCH_SIZE=64
LR=0.001
DATASET=cifar100
DATA_DIR="./../../../data/cifar100"
CLIENT_OPTIMIZER=adam
BACKEND=TRPC
TRPC_MASTER_CONFIG_PATH="./communication_benchmark/trpc/trpc_master_config.csv"
CI=0

date=$(date +%s)
logfile="rpc_InfiniBand_enabled.$date.log"
PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

echo "Using _transport cuda_gdr" >> ./$logfile

(cd ../.. && mpirun -np $PROCESS_NUM -hostfile ./communication_benchmark/trpc/mpi_host_file python3 ./main_fedavg.py \
  --gpu_mapping_file "./communication_benchmark/trpc/gpu_mapping.yaml" \
  --gpu_mapping_key "mapping_FedML_tRPC" \
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
  --enable_cuda_rpc \
  --ci $CI \
  --trpc_master_config_path $TRPC_MASTER_CONFIG_PATH
) 
# >> ./$logfile 2>&1
  
