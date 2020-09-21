#!/usr/bin/env bash



ROOT=$1
RUN_ID=$2
GROUP_ID=$3
MODE=$4
LR=$5
N=$6
NHB_NUM_U=$7
NHB_NUM_D=$8
B_SYMMETRIC=$9

python3 main_dol.py \
--root_path $ROOT \
--run_id $RUN_ID \
--group_id $GROUP_ID \
--mode $MODE \
--learning_rate $LR \
--client_number $N \
--topology_neighbors_num_undirected $NHB_NUM_U \
--topology_neighbors_num_directed $NHB_NUM_D \
--b_symmetric $B_SYMMETRIC \
--data_name "RO" \
--beta 0.5 \
--iteration_number 1000 \
--epoch 2