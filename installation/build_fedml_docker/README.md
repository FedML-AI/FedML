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

ARCH="x86_64"
OS="ubuntu20.04"
DISTRO="ubuntu2004"
PYTHON_VERSION="3.8"
PYTORCH_VERSION="1.13.1"
NCCL_VERSION="2.11.4"
CUDA_VERSION="11.6"
LIB_NCCL="2.11.4-1+cuda11.6"
NVIDIA_BASE_IMAGE="nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04"
PYTORCH_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu116"
PYTORCH_GEOMETRIC_URL="https://data.pyg.org/whl/torch-1.13.1+cu116.html"
OUTPUT_IMAGE="fedml/fedml:latest-torch1.13.1-cuda11.6-cudnn8-devel"
  
bash build-docker.sh $ARCH $OS $DISTRO $PYTHON_VERSION $PYTORCH_VERSION $NCCL_VERSION $CUDA_VERSION \
     $OUTPUT_IMAGE $NVIDIA_BASE_IMAGE $PYTORCH_EXTRA_INDEX_URL $PYTORCH_GEOMETRIC_URL $LIB_NCCL
```

## Build Docker Image (arm64v8)
```
# https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-12-10.html#rel_2-12-10

ARCH="arm64"
OS="ubuntu20.04"
DISTRO="ubuntu2004"
PYTHON_VERSION="3.8"
PYTORCH_VERSION="1.13.1"
NCCL_VERSION="2.11.4"
CUDA_VERSION="11.6"
LIB_NCCL="2.11.4-1+cuda11.6"
NVIDIA_BASE_IMAGE="nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04"
PYTORCH_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu116"
PYTORCH_GEOMETRIC_URL="https://data.pyg.org/whl/torch-1.13.1+cu116.html"
OUTPUT_IMAGE="fedml/fedml:latest-arm64-torch1.13.1-cuda11.6-cudnn8-devel"
  
bash build-docker.sh $ARCH $OS $DISTRO $PYTHON_VERSION $PYTORCH_VERSION $NCCL_VERSION $CUDA_VERSION \
     $OUTPUT_IMAGE $NVIDIA_BASE_IMAGE $PYTORCH_EXTRA_INDEX_URL $PYTORCH_GEOMETRIC_URL $LIB_NCCL
```

## Push docker to the cloud (change image-id to your own)

```
# c56e5f90d546 is the docker image ID
docker tag c56e5f90d546 fedml/fedml:latest-torch1.13.1-cuda11.6-cudnn8-devel
docker login --username fedml
docker push fedml/fedml:latest-torch1.13.1-cuda11.6-cudnn8-devel
```
