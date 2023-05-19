#!/usr/bin/env bash
set -e

# see https://stackoverflow.com/a/17841619
function join_by {
  local d=${1-} f=${2-}
  if shift 2; then
    printf %s "$f" "${@/#/$d}"
  fi
}

BASE_DIR="$(dirname "$0")"
BASE_DIR="$(realpath "${BASE_DIR}/../../")"
cd "${BASE_DIR}"

TARGET="${1:-"client"}"
CONFIG_PATH="${2:-"mlops_config"}" # must be the directory containing the config file

GIT_IGNORE_PATTERN=(
  "cmake-build-*"
  ".idea"
  ".vscode"
  "*.DS_Store"
  "node_modules"
  "package-lock.json"
  "__pycache__"
  "*.pyc"
  "*.egg-info"
  "build"
  "dist"
  ".ipynb_checkpoints"
  ".python-version"
  "wandb"
  # project files
  "*_host_file"
  ".data"
  ".logs"
  "cache"
  "devops/geo-distributed-cluster"
)
IGNORE_PATTERN=(
  "${GIT_IGNORE_PATTERN[@]}"
  "scripts/*"
  "build"
  "assets"
  ".python-version"
  "*.zip"
  "*.backup.*"
  "*debug*"
  # specific files
  "check_ckpt.py"
)

IGNORE_STR="$(join_by "," "${IGNORE_PATTERN[@]}")"
#echo "${IGNORE_STR}"

fedml build \
  -t "${TARGET}" \
  -sf . \
  -ep main_mlops.py \
  -df build \
  -cf "${CONFIG_PATH}" \
  -ig "${IGNORE_STR}"
