#!/bin/bash
set -x
# sudo kill $(ps aux | grep "deepspeed_ep_train.py" | grep -v grep | awk '{print $2}')

WORKSPACE=$1
NODE_NUM=$2
MASTER_IP=$3
HOST_FILE_FOR_PDSH=$4

echo $NODE_NUM

cat $HOST_FILE_FOR_PDSH

ip=$(/sbin/ip -o -4 addr list eth0 | awk '{print $4}' | cut -d/ -f1)
echo $ip
index=$(grep -n $ip $HOST_FILE_FOR_PDSH | cut -d : -f 1)
((index-=1))
echo $index


# install nvidia-docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | sudo tee /etc/yum.repos.d/nvidia-docker.repo
sudo yum clean expire-cache
sudo yum install -y nvidia-docker2
sudo systemctl restart docker

if [ $index -ge 0 ]
then
  sleep 5
else
  echo "master node"
fi

sudo chmod 777 /var/run/docker.sock

cat ./docker/password.txt | docker login --username fedml --password-stdin

echo "stop previous docker run..."
#docker rm $(docker ps -aq)
docker container kill $(docker ps -q)


nvidia-docker run -i -v $WORKSPACE:/$WORKSPACE --shm-size=60g --ulimit nofile=65535 --ulimit memlock=-1 --privileged \
--env AWS_BATCH_JOB_NODE_INDEX=$index \
--env AWS_BATCH_JOB_NUM_NODES=$NODE_NUM \
--env AWS_BATCH_JOB_MAIN_NODE_INDEX=0 \
--env AWS_BATCH_JOB_ID=string \
--env AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS=$MASTER_IP \
--env  \
FEDML_BATCH_BOOTSTRAP=$WORKSPACE/scripts/cross-silo/docker/boostrap.sh \
--env \
FEDML_BATCH_ENTRY_SCRIPT=$WORKSPACE/scripts/cross-silo/docker/entry_script.sh \
-u fedml --net=host \
fedml/fedml:1.2
