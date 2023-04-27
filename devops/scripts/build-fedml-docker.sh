version=base
pwd=`pwd`
build_arm_arch_images=$1

export FEDML_VERSION=`cat python/setup.py |grep version= |awk -F'=' '{print $2}' |awk -F',' '{print $1}'|awk -F'"' '{print $2}'`
echo "version from setup.py: ${FEDML_VERSION}"

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

if [ "$build_arm_arch_images" = "" ]; then
  # Build X86_64 docker
  ARCH="x86_64"

  OUTPUT_IMAGE="fedml/fedml:latest-torch1.13.1-cuda11.6-cudnn8-devel"
  CURRENT_IMAGE="fedml/fedml:${FEDML_VERSION}-torch1.13.1-cuda11.6-cudnn8-devel"

  cd ./installation/build_fedml_docker
  docker rmi $OUTPUT_IMAGE
  docker rmi $CURRENT_IMAGE
  bash build-docker.sh $ARCH $OS $DISTRO $PYTHON_VERSION $PYTORCH_VERSION $NCCL_VERSION $CUDA_VERSION \
       $OUTPUT_IMAGE $NVIDIA_BASE_IMAGE $PYTORCH_EXTRA_INDEX_URL $PYTORCH_GEOMETRIC_URL $LIB_NCCL

  docker tag $OUTPUT_IMAGE $CURRENT_IMAGE
  cd $pwd
fi

if [ "$build_arm_arch_images" != "" ]; then
  # Build ARM_64 docker
  ARCH="arm64"
  OUTPUT_IMAGE="fedml/fedml:latest-arm64-torch1.13.1-cuda11.6-cudnn8-devel"
  NVIDIA_BASE_IMAGE="nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04@sha256:1a06a6cc47ba6ade96c646231c3d0f3216f9b32fb1420f88e46616eea478a661"
  CURRENT_IMAGE="fedml/fedml:${FEDML_VERSION}-arm64-torch1.13.1-cuda11.6-cudnn8-devel"

  cd ./installation/build_fedml_docker
  docker rmi $OUTPUT_IMAGE
  docker rmi $CURRENT_IMAGE
  bash build-docker.sh $ARCH $OS $DISTRO $PYTHON_VERSION $PYTORCH_VERSION $NCCL_VERSION $CUDA_VERSION \
       $OUTPUT_IMAGE $NVIDIA_BASE_IMAGE $PYTORCH_EXTRA_INDEX_URL $PYTORCH_GEOMETRIC_URL $LIB_NCCL

  docker tag $OUTPUT_IMAGE $CURRENT_IMAGE

  cd $pwd

  # Build nvidia_jetson docker
  ARCH="jetson"
  OUTPUT_IMAGE="fedml/fedml:latest-nvidia-jetson-l4t-ml-r35.1.0-py3"
  NVIDIA_BASE_IMAGE="nvidia/cuda:11.6.1-cudnn8-devel-ubuntu18.04@sha256:1a06a6cc47ba6ade96c646231c3d0f3216f9b32fb1420f88e46616eea478a661"
  CURRENT_IMAGE="fedml/fedml:${FEDML_VERSION}-nvidia-jetson-l4t-ml-r35.1.0-py3"

  cd ./installation/build_fedml_docker
  docker rmi $OUTPUT_IMAGE
  docker rmi $CURRENT_IMAGE
  bash build-docker.sh $ARCH $OS $DISTRO $PYTHON_VERSION $PYTORCH_VERSION $NCCL_VERSION $CUDA_VERSION \
       $OUTPUT_IMAGE $NVIDIA_BASE_IMAGE $PYTORCH_EXTRA_INDEX_URL $PYTORCH_GEOMETRIC_URL $LIB_NCCL

  docker tag $OUTPUT_IMAGE $CURRENT_IMAGE

  cd $pwd

  # Build rpi32 docker
  #ARCH="rpi32"
  #OUTPUT_IMAGE="fedml/fedml:latest-raspberrypi4-32-py38"
  #NVIDIA_BASE_IMAGE="nvidia/cuda:11.6.1-cudnn8-devel-ubuntu18.04@sha256:1a06a6cc47ba6ade96c646231c3d0f3216f9b32fb1420f88e46616eea478a661"
  #CURRENT_IMAGE="fedml/fedml:${FEDML_VERSION}-raspberrypi4-32-py38"

  cd ./installation/build_fedml_docker
  docker rmi $OUTPUT_IMAGE
  docker rmi $CURRENT_IMAGE
  bash build-docker.sh $ARCH $OS $DISTRO $PYTHON_VERSION $PYTORCH_VERSION $NCCL_VERSION $CUDA_VERSION \
       $OUTPUT_IMAGE $NVIDIA_BASE_IMAGE $PYTORCH_EXTRA_INDEX_URL $PYTORCH_GEOMETRIC_URL $LIB_NCCL

  docker tag $OUTPUT_IMAGE $CURRENT_IMAGE

  cd $pwd

  # Build rpi64 docker
  ARCH="rpi64"
  OUTPUT_IMAGE="fedml/fedml:latest-raspberrypi4-64-py38"
  NVIDIA_BASE_IMAGE="nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04@sha256:1a06a6cc47ba6ade96c646231c3d0f3216f9b32fb1420f88e46616eea478a661"
  CURRENT_IMAGE="fedml/fedml:${FEDML_VERSION}-raspberrypi4-64-py38"

  cd ./installation/build_fedml_docker
  docker rmi $OUTPUT_IMAGE
  docker rmi $CURRENT_IMAGE
  bash build-docker.sh $ARCH $OS $DISTRO $PYTHON_VERSION $PYTORCH_VERSION $NCCL_VERSION $CUDA_VERSION \
       $OUTPUT_IMAGE $NVIDIA_BASE_IMAGE $PYTORCH_EXTRA_INDEX_URL $PYTORCH_GEOMETRIC_URL $LIB_NCCL

  docker tag $OUTPUT_IMAGE $CURRENT_IMAGE

  cd $pwd
fi
