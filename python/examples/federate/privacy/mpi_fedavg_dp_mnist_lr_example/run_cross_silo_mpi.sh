#!/usr/bin/env bash

WORKER_NUM=$1

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

mpirun -np $PROCESS_NUM \
-hostfile mpi_host_file --oversubscribe \
python cross_silo_with_mpi.py --cf config/nbafl/fedml_config.yaml