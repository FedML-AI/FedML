#!/usr/bin/env bash

WORKER_NUM=$1

TASK_NAMES=(
"nlp_benign"
"nlp_byzantine_random_1adv_krum_m5"
"nlp_byzantine_random_1adv"
"nlp_byzantine_random_1adv_foolsgold"
"nlp_byzantine_random_1adv_rfa"
)  

IDXs=("seed100")

DATA_PARTITION_TYPE="hetero"

for IDX in "${IDXs[@]}"
do
    for TASK_NAME in "${TASK_NAMES[@]}"
    do
        LOG_FILE="logs/NLP/${WORKER_NUM}clients_${DATA_PARTITION_TYPE}/${DATA_PARTITION_TYPE}_RNN_${TASK_NAME}_${WORKER_NUM}clients_${IDX}.log"

        PROCESS_NUM=`expr $WORKER_NUM + 1`
        echo $PROCESS_NUM

        hostname > mpi_host_file

        CONFIG="config/NLP/${WORKER_NUM}clients_${DATA_PARTITION_TYPE}/${TASK_NAME}.yaml"

        mpirun -np $PROCESS_NUM -hostfile mpi_host_file --oversubscribe \
        python torch_mpi_nlp.py --cf $CONFIG > $LOG_FILE 2>&1
    done
done