#!/usr/bin/env bash
RANK=$1
/home/ubuntu/fednlp_migration/bin/python3.8 -m fednlp.text_classification.client.torch_client --cf fednlp/text_classification/config/fedml_cross_silo_config.yaml --rank $RANK
