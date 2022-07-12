#!/usr/bin/env bash
RANK=$1
/home/ubuntu/fednlp_migration/bin/python3.8 -m fednlp.span_extraction.torch_client --cf fednlp/span_extraction/config/fedml_cross_silo_config.yaml --rank $RANK
