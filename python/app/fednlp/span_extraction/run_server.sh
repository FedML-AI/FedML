#!/usr/bin/env bash
/home/ubuntu/fednlp_migration/bin/python3.8 -m fednlp.span_extraction.torch_server --cf fednlp/span_extraction/config/fedml_cross_silo_config.yaml --rank 0

