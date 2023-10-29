#!/usr/bin/env bash
RANK=$1
python3 main_fedml_chestxray_clf.py --cf config/fedml_config.yaml --run_id healthcare_chestxray --rank $RANK --role client