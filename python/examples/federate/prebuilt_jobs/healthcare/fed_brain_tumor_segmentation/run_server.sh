#!/usr/bin/env bash
RUN_ID=$1
python3 tf_server.py --cf config/fedml_config.yaml --rank 0 --run_id $RUN_ID