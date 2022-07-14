#!/usr/bin/env bash
RANK=$1
python main_fedml_image_classification.py --cf config/fedml_config.yaml --rank $RANK --role client