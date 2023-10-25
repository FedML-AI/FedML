#!/bin/bash

ROLE=$1
ACCOUNT_ID=$2
FEDML_VERSION=release
FEDML_DEVICE_ID=0
FEDML_OS_NAME=linux

if [ "${ROLE}" == "client" ]; then
  python3 ./fedml-pip/fedml/computing/scheduler/master/server_daemon.py -t login -u ${ACCOUNT_ID} -v ${FEDML_VERSION} -r client -id ${FEDML_DEVICE_ID} -os ${FEDML_OS_NAME}
else
  python3 ./fedml-pip/fedml/computing/scheduler/master/server_daemon.py -t login -u ${ACCOUNT_ID} -v ${FEDML_VERSION} -r edge_server -id ${FEDML_DEVICE_ID} -os ${FEDML_OS_NAME}
fi

cur_loop=1
while [ $cur_loop -eq 1 ]
do
  sleep 10
done
