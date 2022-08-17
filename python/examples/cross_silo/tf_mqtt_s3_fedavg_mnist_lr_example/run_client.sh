#!/usr/bin/env bash
RANK=$1
python3 tf_client.py --cf config/fedml_config.yaml --rank $RANK --role client