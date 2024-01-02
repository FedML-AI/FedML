set -e

BASE_DIR="$(dirname "$0")"
BASE_DIR="$(realpath "${BASE_DIR}/../")"
cd "${BASE_DIR}"

# download dataset
bash scripts/setup.sh

### don't modify this part ###
echo "[FedML]Bootstrap Finished"
##############################
