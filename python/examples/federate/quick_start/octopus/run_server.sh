#!/usr/bin/env bash
RUN_ID=$1
python server/torch_server.py --cf config/fedml_config.yaml --role server --rank 0 --run_id $RUN_ID
