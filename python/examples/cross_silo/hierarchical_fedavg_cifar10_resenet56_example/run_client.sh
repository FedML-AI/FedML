#!/usr/bin/env bash
NETWORK_INTERFACE=lo
export NCCL_SOCKET_IFNAME=$NETWORK_INTERFACE
export GLOO_SOCKET_IFNAME=$NETWORK_INTERFACE

RANK=$1
python3 torch_client.py --cf config/fedml_config.yaml --rank $RANK