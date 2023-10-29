#!/usr/bin/env bash

WORKER_NUM=$1

TASK_NAMES=(
"cifar10_benign"
"cifar10_byzantine_zero_1adv_rfa"
"cifar10_byzantine_flip_1adv_foolsgold"
"cifar10_byzantine_zero_1adv"
"cifar10_byzantine_flip_1adv_krum_m5"
"cifar10_foolsgold"
"cifar10_byzantine_flip_1adv_rfa"
"cifar10_krum_m5"
"cifar10_byzantine_flip_1adv"
"cifar10_label_flipping_3291_10P_adv_foolsgold"
"cifar10_byzantine_random_1adv_foolsgold"
"cifar10_label_flipping_3291_10P_adv_krum_m5"
"cifar10_byzantine_random_1adv_krum_m5"
"cifar10_label_flipping_3291_10P_adv_rfa"
"cifar10_byzantine_random_1adv_rfa"
"cifar10_label_flipping_3291_10P_adv"
"cifar10_byzantine_random_1adv"
"cifar10_rfa"
"cifar10_byzantine_zero_1adv_foolsgold"
"cifar10_byzantine_zero_1adv_krum_m5"
)  

IDXs=("3") # "2" "1")

DATA_PARTITION_TYPE="hetero"

for IDX in "${IDXs[@]}"
do
    for TASK_NAME in "${TASK_NAMES[@]}"
    do
        LOG_FILE="logs/CV/${WORKER_NUM}clients/${DATA_PARTITION_TYPE}/${DATA_PARTITION_TYPE}_resnet20_${TASK_NAME}_${WORKER_NUM}clients_${IDX}.log"

        PROCESS_NUM=`expr $WORKER_NUM + 1`
        echo $PROCESS_NUM $TASK_NAME

        hostname > mpi_host_file

        CONFIG="config/CV/${WORKER_NUM}clients/${DATA_PARTITION_TYPE}/${TASK_NAME}.yaml"

        mpirun -np $PROCESS_NUM -hostfile mpi_host_file --oversubscribe \
        python torch_mpi_cv.py --cf $CONFIG > $LOG_FILE 2>&1
    done
done