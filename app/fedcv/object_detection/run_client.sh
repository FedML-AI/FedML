#!/usr/bin/env bash
RANK=$1
python3 torch_fedml_object_detection_client.py --cf config/fedml_object_detection.yaml --rank $RANK