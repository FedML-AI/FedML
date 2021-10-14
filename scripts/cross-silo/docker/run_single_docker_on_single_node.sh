#!/bin/bash

WORKSPACE=$1
MASTER_IP=$2

sudo chmod 777 /var/run/docker.sock

cat ./docker/password.txt | docker login --username fedml --password-stdin

echo "stop previous docker run..."
#docker rm $(docker ps -aq)
#docker container kill $(docker ps -q)


docker run -i -v $WORKSPACE:/$WORKSPACE --shm-size=60g --ulimit nofile=65535 --ulimit memlock=-1 --privileged \
--env AWS_BATCH_JOB_NODE_INDEX=0 \
--env AWS_BATCH_JOB_NUM_NODES=1 \
--env AWS_BATCH_JOB_MAIN_NODE_INDEX=0 \
--env AWS_BATCH_JOB_ID=string \
--env AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS=$MASTER_IP \
--env  \
FEDML_BATCH_BOOTSTRAP=$WORKSPACE/scripts/cross-silo/docker/boostrap.sh \
--env \
FEDML_BATCH_ENTRY_SCRIPT=$WORKSPACE/scripts/cross-silo/docker/entry_script.sh \
-u fedml --net=host \
fedml/fedml:1.2
