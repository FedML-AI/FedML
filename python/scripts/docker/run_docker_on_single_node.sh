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

nvidia-docker run -i -v /fsx-dev:/fsx-dev -v /fsx:/fsx -v /job:/job --shm-size=300g --ulimit nofile=65535 --ulimit memlock=-1 --privileged \
--env AWS_BATCH_JOB_NODE_INDEX=$index \
--env AWS_BATCH_JOB_NUM_NODES=$node_num_for_training \
--env AWS_BATCH_JOB_MAIN_NODE_INDEX=0 \
--env AWS_BATCH_JOB_ID=string \
--env AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS=$master_ip \
--env  \
M5_BATCH_BOOTSTRAP=/fsx-dev/hchaoyan/home/m5/src/MoE-Pretraining/scripts/launcher/4_aws_batch/entry_scripts/p4dn/switch_hi_asg/bootstrap.sh \
--env \
M5_BATCH_ENTRY_SCRIPT=/fsx-dev/hchaoyan/home/m5/src/MoE-Pretraining/scripts/launcher/4_aws_batch/entry_scripts/p4dn/switch_hi_asg/entry_script.sh \
-u deepspeed --net=host \
--device=/dev/infiniband/uverbs0 \
--device=/dev/infiniband/uverbs1 \
--device=/dev/infiniband/uverbs2 \
--device=/dev/infiniband/uverbs3 \
350694149704.dkr.ecr.$REGION.amazonaws.com/m5deepspeed:moe