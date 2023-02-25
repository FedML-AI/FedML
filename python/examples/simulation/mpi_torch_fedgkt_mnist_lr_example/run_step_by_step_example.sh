#!/usr/bin/env bash

WORKER_NUM=$1

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

mpirun -np $PROCESS_NUM \
-hostfile mpi_host_file --oversubscribe \
python torch_fedgkt_cifar10_resnet56.py --cf config/fedml_config.yaml