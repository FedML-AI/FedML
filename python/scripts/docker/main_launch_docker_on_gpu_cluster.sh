#!/bin/bash

# usage:
# nohup sh main_launch_docker_on_gpu_cluster.sh hostfile_pdsh us-east-1 8 > log/p3dn_8_nodes.log 2>&1 &
# nohup sh main_launch_docker_on_gpu_cluster.sh hostfile_pdsh us-east-1 16 > log/p4dn_16_nodes.log 2>&1 &


set -x

# ENV (please change this to yours)
HOST_FILE_FOR_PDSH=$1
REGION=$2
NODE_NUM=$3 # how many nodes to run distributed training
WORKSPACE=/home/chaoyanghe/sourcecode/FedML_startup/FedML

cat $HOST_FILE_FOR_PDSH # only for PDSH (parallel distributed SSH); if you don't have hostfile_pdsh, please check the scripts at "initialize_asg_nodes.sh"
# cat /job/hostfile # this file will be used by DeepSpeed

# scp /job/hostfile to all other machines. "/job/hostfile" wi
MASTER_IP=
while read ip
do
    echo $USER@$ip
    if [ -z "$MASTER_IP" ]
    then
      MASTER_IP=$ip
      echo "\nMASTER_IP is $ip"
    fi
#    scp $HOST_FILE_FOR_PDSH $USER@$ip:"$WORKSPACE/M5Transformers/scripts/launcher/3_asg_multinode_docker/$HOST_FILE_FOR_PDSH"
done < $HOST_FILE_FOR_PDSH

echo $USER@$ip
#scp $HOST_FILE_FOR_PDSH $USER@$ip:"$WORKSPACE/M5Transformers/scripts/launcher/3_asg_multinode_docker/$HOST_FILE_FOR_PDSH"

echo "\nMASTER_IP is $MASTER_IP"

pdsh -w ^$HOST_FILE_FOR_PDSH -R ssh "sudo pkill python; \
cd $WORKSPACE/scripts/launcher/3_asg_multinode_docker; \
sh run_docker_on_single_node.sh $REGION $NODE_NUM $MASTER_IP $HOST_FILE_FOR_PDSH"