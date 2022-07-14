#!/usr/bin/env bash
RANK=$1

fedml launch main_fedml_cross_silo_hi.py --cf config/fedml_config.yaml --rank $RANK --role client