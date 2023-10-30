#!/usr/bin/env bash
RANK=$1
python3 main_fedml_tcga_brca.py --cf config/fedml_config.yaml --run_id tcga_brca --rank $RANK --role client