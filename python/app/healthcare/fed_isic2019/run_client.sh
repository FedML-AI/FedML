#!/usr/bin/env bash
RANK=$1
python3 main_fedml_isic2019.py --cf config/fedml_config.yaml --run_id isic2019 --rank $RANK --role client