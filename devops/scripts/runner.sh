#!/bin/bash --login
# The --login ensures the bash configuration is loaded,
# enabling Conda.

ACCOUNT_ID=$1
FEDML_VERSION=$2
FEDML_RUNNER_CMD=$3

# Enable strict mode.
set -euo pipefail
# ... Run whatever commands ...

# Temporarily disable strict mode and activate conda:
set +euo pipefail
conda activate fedml_env

# Re-enable strict mode:
set -euo pipefail

# exec the final command:
exec fedml login ${ACCOUNT_ID} -v ${FEDML_VERSION} -s -r cloud_server -rc ${FEDML_RUNNER_CMD}

cur_loop=1
while [ $cur_loop -eq 1 ]
do
  sleep 10
done
