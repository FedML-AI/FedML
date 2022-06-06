# This folder maintains the docker-related scripts

## Preparation
1. install docker engine according to https://docs.docker.com/engine/install/ubuntu/

2. install nvidia-docker tools.
```
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
sudo chmod 777 /var/run/docker.sock

```

3. Setting Up ARM Emulation on x86 (optional)

https://www.stereolabs.com/docs/docker/building-arm-container-on-x86/

```
sudo apt-get install qemu binfmt-support qemu-user-static
```

## Build Docker Image (x86_64)
```
# https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-12-10.html#rel_2-12-10

ARCH=x86_64
OS=ubuntu20.04
DISTRO=ubuntu2004
PYTHON_VERSION=3
PYTORCH_VERSION=1.11.0
NCCL_VERSION=2.11.4
CUDA_VERSION=11.4
bash build-docker.sh $ARCH $OS $DISTRO $PYTHON_VERSION $PYTORCH_VERSION $NCCL_VERSION $CUDA_VERSION
```

## Build Docker Image (arm64v8)
```
# https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-12-10.html#rel_2-12-10

ARCH=arm64v8
OS=ubuntu20.04
DISTRO=ubuntu2004
PYTHON_VERSION=3
PYTORCH_VERSION=1.11.0
NCCL_VERSION=2.11.4
CUDA_VERSION=11.4
bash build-docker.sh $ARCH $OS $DISTRO $PYTHON_VERSION $PYTORCH_VERSION $NCCL_VERSION $CUDA_VERSION
```

## Push docker to the cloud (change image-id to your own)

```
# c56e5f90d546 is the docker image ID
docker tag c56e5f90d546 fedml/fedml:cuda-11.4.0-devel-ubuntu20.04
docker login --username fedml
docker push fedml/fedml:cuda-11.4.0-devel-ubuntu20.04
```
