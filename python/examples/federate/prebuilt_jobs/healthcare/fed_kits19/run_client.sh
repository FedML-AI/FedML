#!/usr/bin/env bash
RANK=$1
python3 main_fedml_kits19.py --cf config/fedml_config.yaml --run_id kits19 --rank $RANK --role client