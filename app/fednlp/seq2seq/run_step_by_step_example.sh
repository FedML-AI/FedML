#!/usr/bin/env bash
WORKER_NUM=$1

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

$(which mpirun) -np $PROCESS_NUM \
-hostfile mpi_host_file \
/home/ubuntu/fedml/bin/python3.8 -m fednlp.seq2seq.torch_fedavg_20news_bert_step_by_step_example --cf fednlp/seq2seq/config/fedml_config.yaml
