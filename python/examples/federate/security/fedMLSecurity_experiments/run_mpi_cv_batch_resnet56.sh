#!/usr/bin/env bash

WORKER_NUM=$1

TASK_NAMES=(
"cifar100_benign"
"cifar100_byzantine_random_1adv_krum_m5"
"cifar100_byzantine_random_1adv"
"cifar100_byzantine_random_1adv_foolsgold"
"cifar100_byzantine_random_1adv_rfa"
)  

IDXs=("5" "4")

DATA_PARTITION_TYPE="hetero"

for IDX in "${IDXs[@]}"
do
    for TASK_NAME in "${TASK_NAMES[@]}"
    do
        LOG_FILE="logs/CV/resnet56_${WORKER_NUM}clients/${DATA_PARTITION_TYPE}_resnet56_${TASK_NAME}_${WORKER_NUM}clients_${IDX}.log"

        PROCESS_NUM=`expr $WORKER_NUM + 1`
        echo $PROCESS_NUM $TASK_NAME

        hostname > mpi_host_file

        CONFIG="config/CV/resnet56_${WORKER_NUM}clients_${DATA_PARTITION_TYPE}/${TASK_NAME}.yaml"

        mpirun -np $PROCESS_NUM -hostfile mpi_host_file --oversubscribe \
        python torch_mpi_cv_resnet56.py --cf $CONFIG > $LOG_FILE 2>&1
    done
done