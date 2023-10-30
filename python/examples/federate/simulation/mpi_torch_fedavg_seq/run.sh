#!/usr/bin/env bash

WORKER_NUM=$1
MPI_HOST=$2

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

# hostname > mpi_host_file

echo $PROCESS_NUM
echo $MPI_HOST

mpirun -np $PROCESS_NUM \
-host $MPI_HOST \
python torch_fedavg.py --cf config/fedml_config.yaml
