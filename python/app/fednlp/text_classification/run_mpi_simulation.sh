#!/usr/bin/env bash
WORKER_NUM=$1

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

$(which mpirun) -np $PROCESS_NUM \
-hostfile mpi_host_file \
/home/ubuntu/fednlp_migration/bin/python3.8 -m fednlp.text_classification.torch_mpi_simulation --cf fednlp/text_classification/config/fedml_config.yaml
