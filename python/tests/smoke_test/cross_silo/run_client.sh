#!/usr/bin/env bash
RANK=$1
RUN_ID=$2
python3 client/torch_client.py --cf config/fedml_config.yaml --rank $RANK --run_id $RUN_ID