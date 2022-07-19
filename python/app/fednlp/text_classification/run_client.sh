#!/usr/bin/env bash
RANK=$1
python torch_main.py --cf config/fedml_config.yaml --rank $RANK --role client
