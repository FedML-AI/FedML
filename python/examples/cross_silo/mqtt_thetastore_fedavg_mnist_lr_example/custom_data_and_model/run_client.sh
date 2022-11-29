#!/usr/bin/env bash
RANK=$1
python3 torch_client.py --cf config/fedml_config.yaml --rank $RANK --role client --run_id theta_test_id