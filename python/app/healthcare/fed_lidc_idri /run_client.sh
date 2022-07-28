#!/usr/bin/env bash
RANK=$1
python3 main_fedml_lidc_idri.py --cf config/fedml_config.yaml --run_id lidc_idri --rank $RANK --role client