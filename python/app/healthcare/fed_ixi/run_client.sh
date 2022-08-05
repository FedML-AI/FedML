#!/usr/bin/env bash
RANK=$1
python3 main_fedml_ixi.py --cf config/fedml_config.yaml --run_id ixi --rank $RANK --role client