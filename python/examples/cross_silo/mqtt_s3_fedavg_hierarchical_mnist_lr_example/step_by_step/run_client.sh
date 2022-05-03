#!/usr/bin/env bash
NETWORK_INTERFACE=lo
export NCCL_SOCKET_IFNAME=$NETWORK_INTERFACE
export GLOO_SOCKET_IFNAME=$NETWORK_INTERFACE

RANK=$1
PROCESS_NUM=2

# $(which mpirun) -np $PROCESS_NUM python3 torch_client.py --cf config/fedml_config.yaml --rank $RANK
# torchrun --standalone --nnodes=1 --nproc_per_node=$PROCESS_NUM torch_client.py --cf config/fedml_config.yaml --rank $RANK
# python3 torch_client.py --cf config/fedml_config.yaml --rank $RANK
python3 client_dist_launcher.py --cf config/fedml_config.yaml --rank $RANK