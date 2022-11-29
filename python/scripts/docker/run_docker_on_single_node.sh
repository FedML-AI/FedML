REGION=$1
node_num_for_training=$2
master_ip=$3
HOST_FILE_FOR_PDSH=$4

echo $node_num_for_training

cat $HOST_FILE_FOR_PDSH

ip=$(/sbin/ip -o -4 addr list eth0 | awk '{print $4}' | cut -d/ -f1)
echo $ip
index=$(grep -n $ip $HOST_FILE_FOR_PDSH | cut -d : -f 1)
((index-=1))
echo $index

if [ $index -ge $node_num_for_training ]
then
  echo "node $index will not join the training"
  exit 0
else
  echo "node $index joins the training"
fi

if [ $index -ge 0 ]
then
  sleep 1
else
  echo "master node"
fi
#--env HOSTNAME $master_ip \

echo "docker...$REGION"

sudo chmod 777 /var/run/docker.sock

# $(aws ecr get-login --no-include-email --region us-east-1)
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin 350694149704.dkr.ecr.$REGION.amazonaws.com

echo "stop previous docker run..."
# docker rm $(docker ps -aq)
docker container kill $(docker ps -q)

echo "start new docker run"
# https://us-east-1.console.aws.amazon.com/ecr/repositories/private/350694149704/m5deepspeed?region=us-east-1

FEDML_DOCKER_IMAGE=fedml/fedml:latest-torch1.12.1-cuda11.3-cudnn8-devel
WORKSPACE=/home/chaoyanghe/sourcecode/FedML_startup/FedML
docker run -t -i -v $WORKSPACE:$WORKSPACE --shm-size=64g --ulimit nofile=65535 --ulimit memlock=-1 --privileged \
--env FEDML_NODE_INDEX=0 \
--env WORKSPACE=$WORKSPACE \
--env FEDML_NUM_NODES=1 \
--env FEDML_MAIN_NODE_INDEX=0 \
--env FEDML_RUN_ID=0 \
--env FEDML_MAIN_NODE_PRIVATE_IPV4_ADDRESS=127.0.0.1 \
--env FEDML_BATCH_BOOTSTRAP=$WORKSPACE/python/scripts/docker/bootstrap.sh \
--env FEDML_BATCH_ENTRY_SCRIPT=$WORKSPACE/python/scripts/docker/entry.sh \
--gpus all \
-u fedml --net=host \
FEDML_DOCKER_IMAGE