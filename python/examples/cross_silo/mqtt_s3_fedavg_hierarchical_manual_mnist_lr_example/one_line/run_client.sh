#!/usr/bin/env bash
RANK=$1
NODE_RANK=$2

python3 client_dist_launcher.py --cf config/fedml_config.yaml --rank $RANK --node_rank $NODE_RANK
