#!/usr/bin/env bash
DATASET=$1

hostname > mpi_host_file

mpirun -np 3 -hostfile ./mpi_host_file python3 ./main_vfl.py \
  --dataset $DATASET \
  --client_number 2 \
  --comm_round 3