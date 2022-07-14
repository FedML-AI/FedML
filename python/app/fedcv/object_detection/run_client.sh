#!/usr/bin/env bash
RANK=$1
python3 main_fedml_object_detection.py --cf config/fedml_config.yaml --rank $RANK --role client