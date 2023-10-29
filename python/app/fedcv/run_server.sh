#!/usr/bin/env bash
RUN_ID=$1

python3 main_fedml_object_detection.py --cf config/cross-silo/fedml_config.yaml --run_id $RUN_ID --rank 0 --role server
