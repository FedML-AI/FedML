#!/usr/bin/env bash

#############################centralized online learning: (COL)###################################
# beta = 0, N = 128, undirected_neighbor_num=128, out_directed_neighbor_num=0
# sh run.sh "./" 115 001 DOL 0.50 128 128 0 1
# 0.4782 (*)

#############################symmetric decentralized online learning (symmetric-DOL ###################################
# beta = 0, N = 128, undirected_neighbor_num=16, out_directed_neighbor_num=0
# sh run.sh "./" 217 002 DOL 0.25 128 16 0 1
# 0.4910 (*)

############################# asymmetric decentralized online learning (asymmetric-DOL)###################
# beta = 0, N = 128, undirected_neighbor_num=16, out_directed_neighbor_num=16
# sh run.sh "./" 314 003 DOL 0.50 128 16 16 0
# 0.4895 (*)

############################asymmetric pushsum-based decentralized online learning (asymmetric-pushsum)#################
# beta = 0, N = 128, undirected_neighbor_num=16, out_directed_neighbor_num=16
# sh run.sh "./" 413 004 PUSHSUM 0.40 128 16 16 0
# 0.4822 (*)

#############################isolation online learning###################################
# beta = 0.5, N = 128, undirected_neighbor_num=128, out_directed_neighbor_num=0
# sh run.sh "./" 1100 011 LOCAL 0.035 128 128 0 1


ROOT=$1
RUN_ID=$2
GROUP_ID=$3
MODE=$4
LR=$5
N=$6
NHB_NUM_U=$7
NHB_NUM_D=$8
B_SYMMETRIC=$9

python3 src/main.py \
--root_path $ROOT \
--run_id $RUN_ID \
--group_id $GROUP_ID \
--mode $MODE \
--learning_rate $LR \
--client_number $N \
--topology_neighbors_num_undirected $NHB_NUM_U \
--topology_neighbors_num_directed $NHB_NUM_D \
--b_symmetric $B_SYMMETRIC \
--data_name "SUSY" \
--beta 0 \
--iteration_number 2000 \
--epoch 1 \
--time_varying 1