#!/usr/bin/env bash
RANK=$1
python3 main_fedml_heart_disease.py --cf config/fedml_config.yaml --run_id heart_disease --rank $RANK --role client