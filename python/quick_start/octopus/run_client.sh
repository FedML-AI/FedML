#!/usr/bin/env bash
RANK=$1
RUN_ID=$2
python client/torch_client.py --cf config/fedml_config.yaml --role client --rank $RANK --run_id $RUN_ID
