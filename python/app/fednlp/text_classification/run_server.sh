#!/usr/bin/env bash
/home/ubuntu/fednlp_migration/bin/python3.8 -m fednlp.text_classification.server.torch_server --cf fednlp/text_classification/config/fedml_cross_silo_config.yaml --rank 0
