#!/usr/bin/env bash

RUN_ID=$1
python3 torch_server.py --cf config/fedml_config.yaml --rank 0 --role server --run_id $RUN_ID