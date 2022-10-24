#!/bin/bash

ARCH=$1
OS=$2
DISTRO=$3
PYTHON_VERSION=$4
PYTORCH_VERSION=$5
NCCL_VERSION=$6
CUDA_VERSION=$7
OUTPUT_IMAGE=$8
NVIDIA_BASE_IMAGE=""
if [ $# == 9 ]; then
  NVIDIA_BASE_IMAGE=$9
fi


DOCKER_FILE_PATH=""
if [[ "$ARCH" == "x86_64" ]]; then
  DOCKER_FILE_PATH=./x86-64/Dockerfile
elif [[  "$ARCH" == "arm64v8" ]]; then
  DOCKER_FILE_PATH=./arm64v8/Dockerfile
elif [[  "$ARCH" == "arm64v8_m1" ]]; then
  DOCKER_FILE_PATH=./arm64v8-apple-m1/Dockerfile
fi

if [ $DOCKER_FILE_PATH == "" ]; then
  echo "Please specify correct arch (choices: x86_64 / arm64v8 /arm64v8_m1)"
  exit -1
fi

if [ $NVIDIA_BASE_IMAGE != "" ]; then
    docker build -f ./x86-64/Dockerfile \
    --build-arg OS=$OS \
    --build-arg DISTRO=$DISTRO \
    --build-arg PYTHON_VERSION=$PYTHON_VERSION \
    --build-arg PYTORCH_VERSION=$PYTORCH_VERSION \
    --build-arg NCCL_VERSION=$NCCL_VERSION \
    --build-arg CUDA_VERSION=$CUDA_VERSION \
    --build-arg NVIDIA_BASE_IMAGE=$NVIDIA_BASE_IMAGE \
    --network=host \
    -t $OUTPUT_IMAGE .
else
    docker build -f ./x86-64/Dockerfile \
    --build-arg OS=$OS \
    --build-arg DISTRO=$DISTRO \
    --build-arg PYTHON_VERSION=$PYTHON_VERSION \
    --build-arg PYTORCH_VERSION=$PYTORCH_VERSION \
    --build-arg NCCL_VERSION=$NCCL_VERSION \
    --build-arg CUDA_VERSION=$CUDA_VERSION \
    --network=host \
    -t $OUTPUT_IMAGE .
fi


