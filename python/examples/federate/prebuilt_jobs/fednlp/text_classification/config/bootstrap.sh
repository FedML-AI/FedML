#!/bin/bash

BASE_DIR="$(dirname "$0")"
BASE_DIR="$(realpath "${BASE_DIR}/../")"

pip install -r "${BASE_DIR}/requirements.txt"
bash "${BASE_DIR}/data/download_data.sh"
bash "${BASE_DIR}/data/download_partition.sh"
