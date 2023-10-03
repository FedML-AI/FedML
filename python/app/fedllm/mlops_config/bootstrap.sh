set -e

BASE_DIR="$(dirname "$0")"
BASE_DIR="$(realpath "${BASE_DIR}/../")"
cd "${BASE_DIR}"

# install dependencies
pip3 install -r requirements.txt

# download datasets
bash scripts/setup.sh "${HOME}/fedml_data/FedLLM"

### don't modify this part ###
echo "[FedML]Bootstrap Finished"
##############################
