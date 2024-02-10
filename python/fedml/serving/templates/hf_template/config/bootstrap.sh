set -e

CURR_DIR="$(pwd)"
BASE_DIR="$(dirname "$0")"
BASE_DIR="$(realpath "${BASE_DIR}/../")"

# Go to base directory
cd "${BASE_DIR}"

ESSENTIALS=(
  "torch>=2.1.0"
  "fedml>=0.8.13"
)

REQUIREMENTS=(
  -r "requirements/requirements.txt"
)

if [[ -z "${GOOGLE_VM_CONFIG_LOCK_FILE}" ]]; then
  # If not on GCP, install extra dependencies
  # see https://github.com/TimDettmers/bitsandbytes/issues/620
  REQUIREMENTS+=(
    -r "requirements/requirements-extra.txt"
  )
fi

echo "Installing essential packages"
pip install "${ESSENTIALS[@]}"

echo "Installing packages for the model"
pip install "${REQUIREMENTS[@]}" "${@}"

# Go back to original directory
cd "${CURR_DIR}"
