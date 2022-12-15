version=base
pwd=`pwd`
build_arm_arch_images=$1

export FEDML_VERSION=`cat python/setup.py |grep version= |awk -F'=' '{print $2}' |awk -F',' '{print $1}'|awk -F'"' '{print $2}'`

if [[ $build_arm_arch_images = "" ]]; then
  # Build X86_64 docker
  ARCH="x86_64"
  OS="ubuntu18.04"
  DISTRO="ubuntu1804"
  PYTHON_VERSION="3.7"
  PYTORCH_VERSION="1.12.1"
  NCCL_VERSION="2.9.9"
  CUDA_VERSION="11.3"
  LIB_NCCL="2.9.9-1+cuda11.3"
  OUTPUT_IMAGE="fedml/fedml:latest-torch1.12.1-cuda11.3-cudnn8-devel"
  NVIDIA_BASE_IMAGE="nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04"
  PYTORCH_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu113"
  PYTORCH_GEOMETRIC_URL="https://data.pyg.org/whl/torch-1.12.0+cu113.html"
  CURRENT_IMAGE="fedml/fedml:${FEDML_VERSION}-torch1.12.1-cuda11.3-cudnn8-devel"

  cd ./docker
  docker rmi $OUTPUT_IMAGE
  docker rmi $CURRENT_IMAGE
  bash build-docker.sh $ARCH $OS $DISTRO $PYTHON_VERSION $PYTORCH_VERSION $NCCL_VERSION $CUDA_VERSION \
       $OUTPUT_IMAGE $NVIDIA_BASE_IMAGE $PYTORCH_EXTRA_INDEX_URL $PYTORCH_GEOMETRIC_URL $LIB_NCCL

  docker tag $OUTPUT_IMAGE $CURRENT_IMAGE
  cd $pwd
fi

if [[ $build_arm_arch_images != "" ]]; then
  # Build ARM_64 docker
  ARCH="arm64"
  OS="ubuntu20.04"
  DISTRO="ubuntu2004"
  PYTHON_VERSION="3.8"
  PYTORCH_VERSION="1.12.1"
  NCCL_VERSION="2.9.6"
  CUDA_VERSION="11.3"
  LIB_NCCL="2.9.6-1+cuda11.3"
  OUTPUT_IMAGE="fedml/fedml:latest-torch1.12.1-cuda11.3-cudnn8-devel-arm64"
  NVIDIA_BASE_IMAGE="nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04@sha256:8e3df8601e81c57e85c082e9bcc6c547641635730ef8516b2cfa9c9e6c1208af"
  PYTORCH_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu113"
  PYTORCH_GEOMETRIC_URL="https://data.pyg.org/whl/torch-1.12.0+cu113.html"
  CURRENT_IMAGE="fedml/fedml:${FEDML_VERSION}-torch1.12.1-cuda11.3-cudnn8-devel-arm64"

  cd ./docker
  docker rmi $OUTPUT_IMAGE
  docker rmi $CURRENT_IMAGE
  bash build-docker.sh $ARCH $OS $DISTRO $PYTHON_VERSION $PYTORCH_VERSION $NCCL_VERSION $CUDA_VERSION \
       $OUTPUT_IMAGE $NVIDIA_BASE_IMAGE $PYTORCH_EXTRA_INDEX_URL $PYTORCH_GEOMETRIC_URL $LIB_NCCL

  docker tag $OUTPUT_IMAGE $CURRENT_IMAGE

  cd $pwd

  # Build nvidia_jetson docker
  ARCH="jetson"
  OS="ubuntu20.04"
  DISTRO="ubuntu2004"
  PYTHON_VERSION="3.7"
  PYTORCH_VERSION="1.12.1"
  NCCL_VERSION="2.9.6"
  CUDA_VERSION="11.3"
  OUTPUT_IMAGE="fedml/fedml:latest-nvidia-jetson-l4t-ml-r35.1.0-py3"
  NVIDIA_BASE_IMAGE="nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04@sha256:8e3df8601e81c57e85c082e9bcc6c547641635730ef8516b2cfa9c9e6c1208af"
  PYTORCH_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu113"
  PYTORCH_GEOMETRIC_URL="https://data.pyg.org/whl/torch-1.12.0+cu113.html"
  CURRENT_IMAGE="fedml/fedml:${FEDML_VERSION}-nvidia-jetson-l4t-ml-r35.1.0-py3"

  cd ./docker
  docker rmi $OUTPUT_IMAGE
  docker rmi $CURRENT_IMAGE
  bash build-docker.sh $ARCH $OS $DISTRO $PYTHON_VERSION $PYTORCH_VERSION $NCCL_VERSION $CUDA_VERSION \
       $OUTPUT_IMAGE $NVIDIA_BASE_IMAGE $PYTORCH_EXTRA_INDEX_URL $PYTORCH_GEOMETRIC_URL $LIB_NCCL

  docker tag $OUTPUT_IMAGE $CURRENT_IMAGE

  cd $pwd

  # Build rpi32 docker
  #ARCH="rpi32"
  #OS="ubuntu20.04"
  #DISTRO="ubuntu2004"
  #PYTHON_VERSION="3.7"
  #PYTORCH_VERSION="1.12.1"
  #NCCL_VERSION="2.9.6"
  #CUDA_VERSION="11.3"
  #OUTPUT_IMAGE="fedml/fedml:latest-raspberrypi4-32-py37"
  #NVIDIA_BASE_IMAGE="nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04@sha256:8e3df8601e81c57e85c082e9bcc6c547641635730ef8516b2cfa9c9e6c1208af"
  #PYTORCH_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu113"
  #PYTORCH_GEOMETRIC_URL="https://data.pyg.org/whl/torch-1.12.0+cu113.html"
  #CURRENT_IMAGE="fedml/fedml:${FEDML_VERSION}-raspberrypi4-32-py37"

  cd ./docker
  docker rmi $OUTPUT_IMAGE
  docker rmi $CURRENT_IMAGE
  bash build-docker.sh $ARCH $OS $DISTRO $PYTHON_VERSION $PYTORCH_VERSION $NCCL_VERSION $CUDA_VERSION \
       $OUTPUT_IMAGE $NVIDIA_BASE_IMAGE $PYTORCH_EXTRA_INDEX_URL $PYTORCH_GEOMETRIC_URL $LIB_NCCL

  docker tag $OUTPUT_IMAGE $CURRENT_IMAGE

  cd $pwd

  # Build rpi64 docker
  ARCH="rpi64"
  OS="ubuntu20.04"
  DISTRO="ubuntu2004"
  PYTHON_VERSION="3.7"
  PYTORCH_VERSION="1.12.1"
  NCCL_VERSION="2.9.6"
  CUDA_VERSION="11.3"
  OUTPUT_IMAGE="fedml/fedml:latest-raspberrypi4-64-py37"
  NVIDIA_BASE_IMAGE="nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04@sha256:8e3df8601e81c57e85c082e9bcc6c547641635730ef8516b2cfa9c9e6c1208af"
  PYTORCH_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu113"
  PYTORCH_GEOMETRIC_URL="https://data.pyg.org/whl/torch-1.12.0+cu113.html"
  CURRENT_IMAGE="fedml/fedml:${FEDML_VERSION}-raspberrypi4-64-py37"

  cd ./docker
  docker rmi $OUTPUT_IMAGE
  docker rmi $CURRENT_IMAGE
  bash build-docker.sh $ARCH $OS $DISTRO $PYTHON_VERSION $PYTORCH_VERSION $NCCL_VERSION $CUDA_VERSION \
       $OUTPUT_IMAGE $NVIDIA_BASE_IMAGE $PYTORCH_EXTRA_INDEX_URL $PYTORCH_GEOMETRIC_URL $LIB_NCCL

  docker tag $OUTPUT_IMAGE $CURRENT_IMAGE

  cd $pwd
fi
