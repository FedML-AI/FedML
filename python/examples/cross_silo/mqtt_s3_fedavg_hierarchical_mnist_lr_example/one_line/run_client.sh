#!/usr/bin/env bash
RANK=$1

python3 client_dist_launcher.py --cf config/fedml_config.yaml --rank $RANK
