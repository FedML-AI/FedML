set -e

verify_download() {
  local file_path="$1"
  local target_md5="$2"
  local file_url="$3"
  local download_dir
  download_dir="$(dirname "${file_path}")"
  local is_download=true

  if [ -f "${file_path}" ]; then
    file_md5="$(md5sum "${file_path}" | cut -d " " -f1)"

    if [[ -z "${target_md5}" ]] || [ "${file_md5}" == "${target_md5}" ]; then
      echo "Found dataset file \"${file_path}\"."
      is_download=false
    fi
  fi

  if [ "${is_download}" == true ]; then
    echo "Dataset file \"${file_path}\" corrupted. Download from remote"
    [ -f "${file_path}" ] && rm "${file_path}"

    mkdir -p "${download_dir}"

    wget --no-check-certificate \
      --content-disposition "${file_url}" \
      --directory-prefix "${download_dir}"
  fi
}

BASE_DIR="$(dirname "$0")"
BASE_DIR="$(realpath "${BASE_DIR}/../")"

DATA_ROOT="${1:-"${BASE_DIR}/.data"}"
DATA_ROOT="$(python3 -c "import os; print(os.path.realpath(os.path.expanduser(\"${DATA_ROOT}\")))")"
mkdir -p "${DATA_ROOT}"

cd "${BASE_DIR}"

TARGET_MD5_LIST=(
  "0378b9f4db7c95332d431c096b1161ee"
  "7dedee20119b86f96cc9fd1544b03905"
  "0f83e20c8ed6fdedf2c69449a7b44d46"
  "2bad82011fab961cd9109ea7e4f444cd"
  "9a75e254bc2659b8eeded73edfacd180"
  "56b34ebfb2b2023deeb682b897f358f5"
  "1e6cc91caefaf740b77230544a90f1a8"
  "71c9b558e990bf9ce14a30e92546fa52"
  "8c7998ceb7a9ee1601ecd3261fe4cb1d"
  "ee3eb3f1d3adb7387862339e25207afc"
  "eada0ca49dc9c92cb95e3e8f2d992e95"
)
DATASET_PATHS=(
  "${DATA_ROOT}/databricks-dolly-15k.jsonl"
  "${DATA_ROOT}/dolly_niid_full/test_databricks-dolly-15k-seed=1234.jsonl"
  "${DATA_ROOT}/dolly_niid_full/train_databricks-dolly-15k-seed=1234.jsonl"
  "${DATA_ROOT}/MedMCQA/train_182822.jsonl"
  "${DATA_ROOT}/MedMCQA/valid_4183.jsonl"
  "${DATA_ROOT}/PubMedQA/test_1000.jsonl"
  "${DATA_ROOT}/PubMedQA/train_211269.jsonl"
  "${DATA_ROOT}/MedQA-USMLE/train_10178.jsonl"
  "${DATA_ROOT}/MedQA-USMLE/valid_1273.jsonl"
  "${DATA_ROOT}/PubMedQA_instruction/test_1000.jsonl"
  "${DATA_ROOT}/PubMedQA_instruction/train_272518.jsonl"
)
DATASET_URLS=(
  "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/d72c16e4644a463b9c678c71d9440befd4594556/databricks-dolly-15k.jsonl"
  "https://fedllm.s3.us-west-2.amazonaws.com/dolly_niid_full/test_databricks-dolly-15k-seed%3D1234.jsonl"
  "https://fedllm.s3.us-west-2.amazonaws.com/dolly_niid_full/train_databricks-dolly-15k-seed%3D1234.jsonl"
  "https://fedllm.s3.us-west-2.amazonaws.com/MedMCQA/train_182822.jsonl"
  "https://fedllm.s3.us-west-2.amazonaws.com/MedMCQA/valid_4183.jsonl"
  "https://fedllm.s3.us-west-2.amazonaws.com/PubMedQA/test_1000.jsonl"
  "https://fedllm.s3.us-west-2.amazonaws.com/PubMedQA/train_211269.jsonl"
  "https://fedllm.s3.us-west-2.amazonaws.com/MedQA-USMLE/train_10178.jsonl"
  "https://fedllm.s3.us-west-2.amazonaws.com/MedQA-USMLE/valid_1273.jsonl"
  "https://fedllm.s3.us-west-2.amazonaws.com/PubMedQA_instruction/test_1000.jsonl"
  "https://fedllm.s3.us-west-2.amazonaws.com/PubMedQA_instruction/train_272518.jsonl"
)

for ((i = 0; i < "${#DATASET_PATHS[@]}"; i++)); do
  verify_download "${DATASET_PATHS[i]}" "${TARGET_MD5_LIST[i]}" "${DATASET_URLS[i]}"
done
