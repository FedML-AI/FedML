#!/usr/bin/env bash
RANK=$1
python3 jax_haiku_client.py --cf config/fedml_config.yaml --rank $RANK --role client --run_id jax_run_example