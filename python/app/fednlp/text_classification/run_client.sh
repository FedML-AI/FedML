#!/usr/bin/env bash
RANK=$1
/home/ubuntu/fednlp_migration/bin/python3.8 torch_client.py --cf config/fedml_config.yaml --rank $RANK
