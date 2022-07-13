#!/usr/bin/env bash
RANK=$1
/home/ubuntu/fednlp_migration/bin/python torch_main.py --cf config/fedml_config.yaml --rank $RANK --role client
