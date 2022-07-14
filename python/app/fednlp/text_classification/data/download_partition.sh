#!/bin/bash

DATA_DIR=~/fednlp_data/
rm -rf ${DATA_DIR}/partition_files
#declare -a data_names=("20news" "agnews" "cnn_dailymail" "cornell_movie_dialogue" "semeval_2010_task8" "sentiment140" "squad_1.1" "sst_2" "ploner" "wikiner" "wmt_cs-en" "wmt_de-en" "wmt_ru-en" "wmt_zh-en" "mrqa" "onto" "gigaword")

# declare -a data_names=("mrqa" "20news" "squad_1.1" "onto" "gigaword" "wikiner" "cornell_movie_dialogue")
declare -a data_names=("20news")

mkdir ${DATA_DIR}
mkdir ${DATA_DIR}/partition_files
for data_name in "${data_names[@]}"
do
	wget --no-check-certificate --no-proxy -P ${DATA_DIR}/partition_files https://fednlp.s3-us-west-1.amazonaws.com/partition_files/${data_name}_partition.h5
done
