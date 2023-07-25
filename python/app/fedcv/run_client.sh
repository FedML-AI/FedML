#!/usr/bin/env bash
RANK=$1
RUN_ID=$2
python3 main_fedml_object_detection.py --cf config/cross-silo/fedml_config.yaml --run_id $RUN_ID --rank $RANK --role client
