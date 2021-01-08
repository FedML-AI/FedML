#!/usr/bin/env bash

WORKER_NUM=$1
MPI_HOST_FILE=$2
PYTHON=$3
ARGS=$4


PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM
echo $MPI_HOST_FILE



mpirun -np $PROCESS_NUM -hostfile ./$MPI_HOST_FILE $PYTHON ./main.py \
  --client_num_per_round $WORKER_NUM \
  $ARGS


