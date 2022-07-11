#!/usr/bin/env bash
RANK=$1
/home/ubuntu/fednlp_migration/bin/python3.8 main_text_classification_client.py --cf config/text_classification_cross_silo_config.yaml --rank $RANK
