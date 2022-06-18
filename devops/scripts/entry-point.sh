#!/bin/bash

ACCOUNT_ID=$1
FEDML_VERSION=$2
FEDML_RUNNER_CMD=$3

echo ${FEDML_RUNNER_CMD}
fedml login ${ACCOUNT_ID} -v ${FEDML_VERSION} -s -r cloud_server -rc ${FEDML_RUNNER_CMD}
./runner.sh
