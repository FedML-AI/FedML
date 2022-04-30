#!/usr/bin/env bash

account_id=$1
version=$2
cur_dir=`pwd`
docker_compose_mode=0

if [ -z ${version} ]; then
  version="release"
fi

export FEDML_IS_USING_AWS_PUBLIC_ECR=1
if [ ${version} != "release" ]; then
  export FEDML_IS_USING_AWS_PUBLIC_ECR=0
fi

if [ $FEDML_IS_USING_AWS_PUBLIC_ECR -eq 1 ]; then
  registry_server=public.ecr.aws
  image_dir=/x6k8q1x9
  client_registry_server=$registry_server
  client_image_dir=$image_dir
else
  registry_server=registry.fedml.ai
  image_dir=/fedml-public-server
  client_registry_server=$registry_server
  client_image_dir=$image_dir
fi
tag=dev
client_tag=dev
if [ ${version} = "local" ]; then
  tag=local
  client_tag=local
fi

usage="Usage: ./run.sh account_id"
eg="eg. ./run.sh 60"

if [ $# != 1 -a $# != 2 ];then
  echo "Please provide two argument."
  echo ${usage};echo ${eg}
  exit -1
fi

echo "Deployment version: "$version

image_path=${image_dir}/fedml-client-agent:${tag}
client_agent_image=$registry_server$image_path
client_base_image=${registry_server}${image_dir}/fedml-cross-silo-cpu:${client_tag}

OS_NAME="Linux"
device_id=""
if [ "$(uname)" == 'Darwin' ]; then
   OS_NAME="MacOS"
   device_id=`system_profiler SPHardwareDataType | grep Serial | awk '{gsub(/ /,"")}{print}' |awk -F':' '{print $2}'`
elif [ "$(expr substr $(uname -s) 1 5)"=="Linux" ]; then
   OS_NAME="Linux"
elif [ "$(expr substr $(uname -s) 1 10)"=="MINGW32_NT" ]; then
   OS_NAME="Windows"
fi
echo "OS Name: ${OS_NAME}"
echo "current dir: $cur_dir"

export FEDML_REGISTER_SERVER=${registry_server}
export FEDML_IMAGE_PATH=${image_path}
export ACCOUNT_ID=${account_id}
export CONFIG_FILE=/fedml/fedml_config/config.yaml
export CURRENT_RUNNING_DIR=${cur_dir}
export CLIENT_VERSION=${client_tag}
export CURRENT_OS_NAME=${OS_NAME}
export CURRENT_DEVICE_ID=${device_id}
export DOCKER_REGISTRY_PUBLIC_SERVER=${client_registry_server}
export DOCKER_REGISTRY_ROOT_DIR=${client_image_dir}

echo "The FedML client agent is being deployed, please wait for a moment..."
docker stop `docker ps -a |grep fedml_container_run_ |grep _edge_ |awk -F' ' '{print $1}'` >/dev/null 2>&1
docker rm `docker ps -a |grep fedml_container_run_ |grep _edge_ |awk -F' ' '{print $1}'` >/dev/null 2>&1
docker stop `docker ps -a |grep fedml-container-run- |grep _edge_ |awk -F' ' '{print $1}'` >/dev/null 2>&1
docker rm `docker ps -a |grep fedml-container-run- |grep _edge_ |awk -F' ' '{print $1}'` >/dev/null 2>&1
docker stop `docker ps -a |grep '/fedml-server/' |awk -F' ' '{print $1}'` >/dev/null 2>&1
docker rm `docker ps -a |grep '/fedml-server/' |awk -F' ' '{print $1}'`  >/dev/null 2>&1
docker rmi `docker ps -a |grep '/fedml-server/' |awk -F' ' '{print $1}'` >/dev/null 2>&1

if [ ${docker_compose_mode} -eq 1 ]; then
  echo "Using docker compose mode"
  docker stop fedml-client-agent >/dev/null 2>&1
  docker rm fedml-client-agent >/dev/null  2>&1
  docker stop fedml-client-agent-docker-client >/dev/null 2>&1
  docker rm fedml-client-agent-docker-client  >/dev/null 2>&1
  docker stop fedml-client-agent-docker-daemon >/dev/null 2>&1
  docker rm fedml-client-agent-docker-daemon  >/dev/null 2>&1
  docker network rm deploy_fedml-client-network >/dev/null 2>&1
  docker volume rm deploy_docker-bin-share >/dev/null 2>&1
  docker-compose down >/dev/null 2>&1
  rm -rf ${cur_dir}/fedml_config >/dev/null  2>&1
  docker pull $client_agent_image >/dev/null 2>&1
  docker-compose up -d --remove-orphans >/dev/null 2>&1
else
  echo "Using docker daemon mode"
  echo "........................."
  docker pull $client_agent_image
  docker stop fedml-client-agent >/dev/null 2>&1
  docker rm fedml-client-agent >/dev/null  2>&1
  rm -rf ${cur_dir}/fedml_config >/dev/null  2>&1

  docker_run_cmd="docker run --name fedml-client-agent "
  if [ ${OS_NAME} == "MacOS" ]; then
    docker_run_cmd="docker run --name fedml-client-agent "
  elif [ ${OS_NAME} == "Linux" ]; then
    docker_run_cmd="docker run --name fedml-client-agent -v $(which  docker):/usr/bin/docker "
  elif [ ${OS_NAME} == "Windows" ]; then
    docker_run_cmd="docker run --name fedml-client-agent "
  fi

  ${docker_run_cmd} \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ${CURRENT_RUNNING_DIR}/fedml_data:/fedml/data \
    -v ${CURRENT_RUNNING_DIR}/fedml_data:/fedml/fedml-package/fedml/data \
    -v ${CURRENT_RUNNING_DIR}/fedml_config:/fedml/conf \
    -v ${CURRENT_RUNNING_DIR}/fedml_run_state:/fedml/fedml_run_state \
    --env ACCOUNT_ID=${ACCOUNT_ID} \
    --env CONFIG_FILE=${CONFIG_FILE} \
    --env CURRENT_RUNNING_DIR=${CURRENT_RUNNING_DIR} \
    --env CLIENT_VERSION=${CLIENT_VERSION} \
    --env OS_NAME=${CURRENT_OS_NAME} \
    --env CURRENT_DEVICE_ID=${CURRENT_DEVICE_ID} \
    --env DOCKER_REGISTRY_PUBLIC_SERVER=${DOCKER_REGISTRY_PUBLIC_SERVER} \
    --env DOCKER_REGISTRY_ROOT_DIR=${DOCKER_REGISTRY_ROOT_DIR} \
    -d $client_agent_image >/dev/null 2>&1
fi

deployed=`docker ps -a |grep fedml-client-agent:${tag} |awk -F'Up' '{print $2}'`
if [ "${deployed}" != "" ]; then
  echo "Congratulations, you have deployed the FedML client agent successfully!"
  device_id_to_display=""
  if [ "${CURRENT_DEVICE_ID}" != "" ]; then
    device_id_to_display=${CURRENT_DEVICE_ID}
  else
    device_id_to_display=`cat ${cur_dir}/fedml_run_state/fedml_client_device_id`
  fi
  echo "Your device id is ${device_id_to_display}. You may review the device in the MLOps edge device list."
  echo "--------------------------------------------------------------------------------------------"
  echo "Now the system will post-process to pull the FedML client docker image to your localhost."
  echo "You may do other things to start your FedML flow. (User Guide: https://doc.fedml.ai)"
  echo "You just need to keep this window not closed until the processing is finished."
  docker pull $client_base_image
  echo "Great, you have succeeded to complete all the running processes."
else
  echo "Oops, you failed to deploy the FedML client agent."
  echo "Please check whether your Docker Application is installed and running normally!"
fi
