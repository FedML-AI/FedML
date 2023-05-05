set -e

BASE_DIR="$(dirname "$0")"
BASE_DIR="$(realpath "${BASE_DIR}/../")"
cd "${BASE_DIR}"

TARGET_DATASET_HASH="0378b9f4db7c95332d431c096b1161ee"

dataset_dir="${BASE_DIR}/.data"
mkdir -p "${dataset_dir}"

dataset_dir="$(realpath "${dataset_dir}")"
dataset_path="${dataset_dir}/databricks-dolly-15k.jsonl"

echo "Preparing dataset"
if [ -f "${dataset_path}" ]; then
  dataset_hash="$(md5sum "${dataset_path}" | cut -d " " -f1)"
else
  dataset_hash=""
fi

if [ "${dataset_hash}" != "${TARGET_DATASET_HASH}" ]; then
  echo "Dataset file \"${dataset_path}\" corrupted. Download from remote"
  [ -f "${dataset_path}" ] && rm "${dataset_path}"

  wget --no-check-certificate \
    --content-disposition "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/d72c16e4644a463b9c678c71d9440befd4594556/databricks-dolly-15k.jsonl" \
    --directory-prefix "${dataset_dir}"
fi
