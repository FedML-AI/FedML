#!/usr/bin/env bash

WORKER_NUM=$1
CONFIG_PATH=$2

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

mpirun -np $PROCESS_NUM \
-hostfile mpi_host_file --oversubscribe \
python torch_step_by_step_example.py --cf $CONFIG_PATH