#!/usr/bin/env bash
WORKER_NUM=$1
PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM
hostname > mpi_host_file
mpirun -np $PROCESS_NUM \
python main_fedml_isic2019.py --cf config/simulation/fedml_config.yaml