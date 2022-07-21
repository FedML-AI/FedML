#!/bin/bash

ARCH=$1
OS=$2
DISTRO=$3
PYTHON_VERSION=$4
PYTORCH_VERSION=$5
NCCL_VERSION=$6
CUDA_VERSION=$7

rm -rf ./FedML
mkdir -p ./FedML
cp -Rf ../* ./FedML/
if [[ "$ARCH" == "x86_64" ]]
thenls

  docker build . -f ./x86-64/Dockerfile \
  --build-arg OS=$OS \
  --build-arg DISTRO=$DISTRO \
  --build-arg PYTHON_VERSION=$PYTHON_VERSION \
  --build-arg PYTORCH_VERSION=$PYTORCH_VERSION \
  --build-arg NCCL_VERSION=$NCCL_VERSION \
  --build-arg CUDA_VERSION=$CUDA_VERSION \
  --network=host

elif [[  "$ARCH" == "arm64v8_m1" ]]
  docker build . -f ./arm64v8-apple-m1/Dockerfile \
  --build-arg OS=$OS \
  --build-arg DISTRO=$DISTRO \
  --build-arg PYTHON_VERSION=$PYTHON_VERSION \
  --build-arg PYTORCH_VERSION=$PYTORCH_VERSION \
  --build-arg NCCL_VERSION=$NCCL_VERSION \
  --build-arg CUDA_VERSION=$CUDA_VERSION \
  --network=host
then
   echo "TBD"
else
   echo "TBD"
fi

