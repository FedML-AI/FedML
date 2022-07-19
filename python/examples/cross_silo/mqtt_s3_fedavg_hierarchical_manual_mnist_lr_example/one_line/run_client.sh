#!/usr/bin/env bash
RANK=$1
NODE_RANK=$2

fedml launch main_fedml_cross_silo_hi.py --cf config/fedml_config.yaml --rank $RANK -node_rank $NODE_RANK --role client