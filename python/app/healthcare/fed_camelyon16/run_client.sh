#!/usr/bin/env bash
RANK=$1
python3 main_fedml_camelyon16.py --cf config/fedml_config.yaml --run_id camelyon16 --rank $RANK --role client