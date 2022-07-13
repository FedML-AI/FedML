#!/usr/bin/env bash

WORKER_NUM=$1
MPI_HOST=$2

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

$(which mpirun) -np $PROCESS_NUM \
-host $MPI_HOST \
python torch_fedavg_mnist_lr_custum_data_and_model_example.py --cf config/zht_config.yaml
