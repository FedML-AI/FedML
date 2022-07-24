#!/usr/bin/env bash
RANK=$1
python3 fedml_iot.py --cf config/fedml_config.yaml --rank $RANK --role client --run_id fediot