#!/usr/bin/env bash
WORKER_NUM=$1

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

$(which mpirun) -np $PROCESS_NUM \
-hostfile mpi_host_file \
/home/ubuntu/fednlp_migration/bin/python3.8 -m fednlp.span_extraction.torch_fedavg_20news_bert_step_by_step_example --cf fednlp/span_extraction/config/fedml_config.yaml
