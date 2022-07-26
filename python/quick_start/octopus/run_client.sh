#!/usr/bin/env bash
RANK=$1
python client/torch_client.py --cf config/fedml_config.yaml --role client --rank $RANK --run_id chaoyang_test
