#!/usr/bin/env bash
RANK=$1
python3 fedml_moleculenet_property_prediction.py --run_id fedgraphnn_cs_ch --cf config/fedml_config.yaml --rank $RANK --role client