#!/usr/bin/env bash
NETWORK_INTERFACE=eno2
export NCCL_SOCKET_IFNAME=$NETWORK_INTERFACE
export GLOO_SOCKET_IFNAME=$NETWORK_INTERFACE

RANK=$1

python3 client_dist_launcher.py --cf config/fedml_config.yaml --rank $RANK
