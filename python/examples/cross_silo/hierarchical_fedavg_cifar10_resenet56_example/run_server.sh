#!/usr/bin/env bash
NETWORK_INTERFACE=lo
export NCCL_SOCKET_IFNAME=$NETWORK_INTERFACE
export GLOO_SOCKET_IFNAME=$NETWORK_INTERFACE

python3 torch_server.py --cf config/fedml_config.yaml --rank 0