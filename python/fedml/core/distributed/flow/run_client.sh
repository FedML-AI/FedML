#!/usr/bin/env bash
RANK=$1

python3 test_fedml_flow.py --cf fedml_config.yaml --rank $RANK
