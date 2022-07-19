#!/usr/bin/env bash
RANK=$1
python3 main_fedml_flamby.py --cf config/fedml_config.yaml --run_id flamby --rank $RANK --role client