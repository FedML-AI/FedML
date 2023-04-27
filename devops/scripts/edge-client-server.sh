#!/bin/bash

ROLE=$1
ACCOUNT_ID=$2
FEDML_VERSION=release
CLIENT_DEVICE_ID=0
CLIENT_OS_NAME=linux

if [[ ${ROLE} == "client" ]]; then
  fedml login ${ACCOUNT_ID} -v ${FEDML_VERSION} -c -id ${CLIENT_DEVICE_ID} -os ${CLIENT_OS_NAME}
else
  CMD fedml login ${ACCOUNT_ID} -v ${FEDML_VERSION} -s -id ${SERVER_DEVICE_ID} -os ${SERVER_OS_NAME}
fi

cur_loop=1
while [ $cur_loop -eq 1 ]
do
  sleep 10
done
