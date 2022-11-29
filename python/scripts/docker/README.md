# Run FedML in Docker

## Run FedML in Docker on a single device

```
docker kill
```

## Run FedML on GPU cluster for high performance simulation

```
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
```