#!/usr/bin/env bash
RANK=$1
python3 fedml_social_networks_clf.py --run_id fedgraphnn_cs_1 --cf config/fedml_config.yaml --rank $RANK --role client