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
if [ $# == 10 ]; then
  PYTORCH_EXTRA_INDEX_URL=$10
else
  PYTORCH_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu113
fi

if [ $# == 11 ]; then
  PYTORCH_GEOMETRIC_URL=$11
else
  PYTORCH_GEOMETRIC_URL=https://data.pyg.org/whl/torch-1.12.0+cu113.html
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
  echo "Please specify the properly arch (options: x86_64 / arm64v8 /arm64v8_m1)"
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
    --build-arg PYTHON_EXTRA_INDEX_URL=$PYTHON_EXTRA_INDEX_URL \
    --build-arg PYTORCH_GEOMETRIC_URL=$PYTORCH_GEOMETRIC_URL \
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
    --build-arg PYTORCH_EXTRA_INDEX_URL=$PYTORCH_EXTRA_INDEX_URL \
    --build-arg PYTORCH_GEOMETRIC_URL=$PYTORCH_GEOMETRIC_URL \
    --network=host \
    -t $OUTPUT_IMAGE .
fi


