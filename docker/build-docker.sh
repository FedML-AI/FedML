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
if [ $# -ge 9 ]; then
  NVIDIA_BASE_IMAGE=$9
fi

if [ $# -ge 10 ]; then
  PYTORCH_EXTRA_INDEX_URL=${10}
else
  PYTORCH_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu113
fi

if [ $# -ge 11 ]; then
  PYTORCH_GEOMETRIC_URL=${11}
else
  PYTORCH_GEOMETRIC_URL=https://data.pyg.org/whl/torch-1.12.0+cu113.html
fi

if [ $# -ge 12 ]; then
  LIB_NCCL=${12}
else
  LIB_NCCL="null"
fi

DOCKER_FILE_PATH=""
if [[ "$ARCH" == "x86_64" ]]; then
  DOCKER_FILE_PATH=./x86-64/Dockerfile
elif [[  "$ARCH" == "arm64" ]]; then
  DOCKER_FILE_PATH=./arm64v8/Dockerfile
elif [[  "$ARCH" == "jetson" ]]; then
  DOCKER_FILE_PATH=./nvidia_jetson/Dockerfile
elif [[  "$ARCH" == "rpi32" ]]; then
  DOCKER_FILE_PATH=./rpi/Dockerfile_32bit_armv7
elif [[  "$ARCH" == "rpi64" ]]; then
  DOCKER_FILE_PATH=./rpi/Dockerfile_64bit_armv8
fi

if [ $DOCKER_FILE_PATH == "" ]; then
  echo "Please specify the properly arch (options: x86_64 / arm64)"
  exit -1
fi

if [[ $NVIDIA_BASE_IMAGE != "" ]]; then
    docker build -f $DOCKER_FILE_PATH \
    --build-arg OS=$OS \
    --build-arg DISTRO=$DISTRO \
    --build-arg PYTHON_VERSION=$PYTHON_VERSION \
    --build-arg PYTORCH_VERSION=$PYTORCH_VERSION \
    --build-arg NCCL_VERSION=$NCCL_VERSION \
    --build-arg CUDA_VERSION=$CUDA_VERSION \
    --build-arg NVIDIA_BASE_IMAGE=$NVIDIA_BASE_IMAGE \
    --build-arg PYTORCH_EXTRA_INDEX_URL=$PYTORCH_EXTRA_INDEX_URL \
    --build-arg PYTORCH_GEOMETRIC_URL=$PYTORCH_GEOMETRIC_URL \
    --build-arg LIB_NCCL=$LIB_NCCL \
    --network=host \
    -t $OUTPUT_IMAGE .
else
    docker build -f $DOCKER_FILE_PATH \
    --build-arg OS=$OS \
    --build-arg DISTRO=$DISTRO \
    --build-arg PYTHON_VERSION=$PYTHON_VERSION \
    --build-arg PYTORCH_VERSION=$PYTORCH_VERSION \
    --build-arg NCCL_VERSION=$NCCL_VERSION \
    --build-arg CUDA_VERSION=$CUDA_VERSION \
    --build-arg PYTORCH_EXTRA_INDEX_URL=$PYTORCH_EXTRA_INDEX_URL \
    --build-arg PYTORCH_GEOMETRIC_URL=$PYTORCH_GEOMETRIC_URL \
    --build-arg LIB_NCCL=$LIB_NCCL \
    --network=host \
    -t $OUTPUT_IMAGE .
fi


