set -e

BASE_DIR="$(dirname "$0")"
BASE_DIR="$(realpath "${BASE_DIR}/../")"
cd "${BASE_DIR}"

TARGET_DATASET_HASH="60afe88200cf8fe9484acc364327a9d0"

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
    --content-disposition "https://github.com/databrickslabs/dolly/raw/master/data/databricks-dolly-15k.jsonl" \
    --directory-prefix "${dataset_dir}"
fi
