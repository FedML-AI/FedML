# Run FedML in Docker

## Run FedML in Docker on a single device

```
docker kill
```

## Run FedML on GPU cluster for high performance simulation

```
WORKSPACE=/home/chaoyanghe/sourcecode/FedML_startup/FedML
nvidia-docker run -t -i -v $WORKSPACE:$WORKSPACE -v --shm-size=64g --ulimit nofile=65535 --ulimit memlock=-1 --privileged \
--env AWS_BATCH_JOB_NODE_INDEX=$index \
--env AWS_BATCH_JOB_NUM_NODES=$node_num_for_training \
--env AWS_BATCH_JOB_MAIN_NODE_INDEX=0 \
--env AWS_BATCH_JOB_ID=string \
--env AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS=$master_ip \
--env  \
BATCH_BOOTSTRAP=/fsx-dev/hchaoyan/home/m5/src/MoE-Pretraining/scripts/launcher/4_aws_batch/entry_scripts/p4dn/switch_hi_asg/bootstrap.sh \
--env \
BATCH_ENTRY_SCRIPT=/fsx-dev/hchaoyan/home/m5/src/MoE-Pretraining/scripts/launcher/4_aws_batch/entry_scripts/p4dn/switch_hi_asg/entry_script.sh \
-u fedml --net=host \
--device=/dev/infiniband/uverbs0 \
--device=/dev/infiniband/uverbs1 \
--device=/dev/infiniband/uverbs2 \
--device=/dev/infiniband/uverbs3 \
fedml/fedml:latest
```