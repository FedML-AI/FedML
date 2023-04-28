#!/bin/bash

ROLE=$1
ACCOUNT_ID=$2
FEDML_VERSION=release
FEDML_DEVICE_ID=0
FEDML_OS_NAME=linux

if [ "${ROLE}" == "client" ]; then
  fedml login ${ACCOUNT_ID} -v ${FEDML_VERSION} -c -id ${FEDML_DEVICE_ID} -os ${FEDML_OS_NAME}
else
  fedml login ${ACCOUNT_ID} -v ${FEDML_VERSION} -s -id ${FEDML_DEVICE_ID} -os ${FEDML_OS_NAME}
fi

cur_loop=1
while [ $cur_loop -eq 1 ]
do
  sleep 10
done
