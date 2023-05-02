#!/usr/bin/env bash
set -e

BASE_DIR="$(dirname "$0")"
BASE_DIR="$(realpath "${BASE_DIR}/../../")"
cd "${BASE_DIR}"

bash scripts/setup.sh

INPUT_PATH="${1:-"${BASE_DIR}/.data/databricks-dolly-15k.jsonl"}"
OUTPUT_PATH="${2:-"${BASE_DIR}/.data/dolly_niid/databricks-dolly-15k.jsonl"}"
CLIENT_NUMBER="${3-2}"
SEED="${4-1234}"
TEST_DATASET_SIZE="${5-1000}"

python3 data/dataset_niid_partition.py \
  --client_number "${CLIENT_NUMBER}" \
  -i "${INPUT_PATH}" \
  -o "${OUTPUT_PATH}" \
  --test_dataset_size "${TEST_DATASET_SIZE}" \
  --seed "${SEED}"
