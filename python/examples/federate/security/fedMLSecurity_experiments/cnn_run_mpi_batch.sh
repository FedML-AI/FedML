#!/usr/bin/env bash

WORKER_NUM=$1

TASK_NAMES=(
"femnist_cnn_byzantine_random_1adv_foolsgold"
"femnist_cnn_byzantine_random_1adv_rfa"
"femnist_cnn_byzantine_random_1adv"
"femnist_cnn_byzantine_random_1adv_krum_m5"
"femnist_cnn_benign"
)  

IDXs=("3" "2" "1")

DATA_PARTITION_TYPE="hetero"

CLIENT_NUM="10"

for IDX in "${IDXs[@]}"
do
    for TASK_NAME in "${TASK_NAMES[@]}"
    do
        LOG_FILE="logs/CNN/${CLIENT_NUM}clients_${DATA_PARTITION_TYPE}/${DATA_PARTITION_TYPE}_CNN_${TASK_NAME}_${CLIENT_NUM}clients_${IDX}.log"

        PROCESS_NUM=`expr $WORKER_NUM + 1`
        echo $PROCESS_NUM $TASK_NAME $IDX

        hostname > mpi_host_file

        CONFIG="config/CNN/${CLIENT_NUM}clients_${DATA_PARTITION_TYPE}/${TASK_NAME}.yaml"

        mpirun -np $PROCESS_NUM -hostfile mpi_host_file --oversubscribe \
        python torch_mpi_cnn.py --cf $CONFIG > $LOG_FILE 2>&1
    done
done