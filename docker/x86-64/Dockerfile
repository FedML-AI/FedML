
# https://hub.docker.com/r/nvidia/cuda/tags
# https://hub.docker.com/r/nvidia/cuda/tags?page=1&name=11.6.0-devel-ubuntu was released at May 6th, 2022
FROM nvidia/cuda:11.4.0-devel-ubuntu20.04
ENV OS=ubuntu20.04
# ARG OS

ENV DISTRO=ubuntu2004
# ARG DISTRO

ENV ARCH=x86_64
# ARG ARCH

ENV PYTHON_VERSION=3
# ARG PYTHON_VERSION

ENV PYTORCH_VERSION=1.11.0
# ARG PYTORCH_VERSION


# https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-12-10.html#rel_2-12-10
ENV NCCL_VERSION=2.11.4
# ARG NCCL_VERSION
# NCCL 2.12.10 release supports CUDA 10.2, CUDA 11.0, and CUDA 11.6.
ENV CUDA_VERSION=11.4
# ARG CUDA_VERSION


RUN echo ${NCCL_VERSION}
RUN echo ${CUDA_VERSION}


ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN export CUDA_HOME=/usr/local/cuda
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
RUN export PATH=$PATH:$CUDA_HOME/bin

ENV OPENMPI_BASEVERSION=4.1
ENV OPENMPI_VERSION=4.1.2

# when building docker on Ubuntu 20.04, we need to fix the issue: NO_PUBKEY A4B469963BF863CC
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
# https://forums.developer.nvidia.com/t/invalid-public-key-for-cuda-apt-repository/212901/20

RUN apt-key del 7fa2af80 \
    && rm /etc/apt/sources.list.d/cuda.list

# ***************************************************************************
# Version and directory Settings
# ***************************************************************************
ENV INSTALL_DIR=/tmp
ENV WORKSPACE=/home/fedml
RUN mkdir -p ${INSTALL_DIR}
RUN mkdir -p ${WORKSPACE}


# ***************************************************************************
# Python
# ***************************************************************************
ARG DEBIAN_FRONTEND=noninteractive
RUN echo "nameserver 8.8.8.8" > /etc/resolv.conf
RUN apt-get clean
RUN apt-get --allow-downgrades update
RUN apt-get install -y python3 python3-pip
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -nsf /usr/bin/pip3 /usr/bin/pip

# ***************************************************************************
# Utilities
# ***************************************************************************
RUN apt-get install -y --no-install-recommends \
        software-properties-common build-essential autotools-dev \
        nfs-common pdsh \
        cmake g++ gcc \
        curl wget vim tmux emacs less unzip \
        htop iftop iotop ca-certificates openssh-client openssh-server \
        rsync iputils-ping net-tools sudo \
        llvm-9-dev

# ***************************************************************************
# Git
# ***************************************************************************
RUN add-apt-repository ppa:git-core/ppa -y && \
    apt-get install -y git && \
    git --version

# ***************************************************************************
# OPENMPI
# ***************************************************************************
RUN cd ${INSTALL_DIR} && \
    wget -q -O - https://download.open-mpi.org/release/open-mpi/v${OPENMPI_BASEVERSION}/openmpi-${OPENMPI_VERSION}.tar.gz | tar xzf - && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --prefix=/usr/local/openmpi-${OPENMPI_VERSION} && \
    make -j"$(nproc)" install && \
    ln -s /usr/local/openmpi-${OPENMPI_VERSION} /usr/local/mpi && \
    # Sanity check:
    test -f /usr/local/mpi/bin/mpic++ && \
    cd ${INSTALL_DIR} && \
    rm -r ${INSTALL_DIR}/openmpi-${OPENMPI_VERSION}
ENV PATH=/usr/local/mpi/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:${LD_LIBRARY_PATH}
# Create a wrapper for OpenMPI to allow running as root by default
RUN mv /usr/local/mpi/bin/mpirun /usr/local/mpi/bin/mpirun.real && \
    echo '#!/bin/bash' > /usr/local/mpi/bin/mpirun && \
    echo 'mpirun.real --allow-run-as-root --prefix /usr/local/mpi "$@"' >> /usr/local/mpi/bin/mpirun && \
    chmod a+x /usr/local/mpi/bin/mpirun


# ***************************************************************************
# NCCL
# ***************************************************************************
# https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html#downloadnccl
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/${ARCH}/7fa2af80.pub && \
add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/${ARCH} /" && \
apt update && \
apt install -y --allow-change-held-packages libnccl2=${NCCL_VERSION}-1+cuda${CUDA_VERSION} libnccl-dev=${NCCL_VERSION}-1+cuda${CUDA_VERSION}

# ***************************************************************************
# PyTorch (install from source)
# ***************************************************************************
RUN sudo pip install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses h5py

RUN cd ${INSTALL_DIR} && \
git clone https://github.com/pytorch/pytorch.git && \
cd pytorch && git checkout v${PYTORCH_VERSION} && \
git submodule sync && \
git submodule update --init --recursive --jobs 0 && \
sudo USE_SYSTEM_NCCL=1 TORCH_CUDA_ARC_LIST="6.0 6.1 7.0 7.5 8.0" CMAKE_CUDA_COMPILER="/usr/local/cuda" python setup.py install
#RUN rm -rf ${INSTALL_DIR}/pytorch
RUN pip install torch==1.11.0

RUN python3 -c "import torch; torch.__version__"

# ***************************************************************************
# install torch-geometric (https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
# ***************************************************************************
RUN cd ${INSTALL_DIR} && \
git clone https://github.com/pyg-team/pytorch_geometric.git && \
cd pytorch_geometric && \
git submodule sync && \
git submodule update --init --recursive --jobs 0 && \
sudo python setup.py install
RUN rm -rf ${INSTALL_DIR}/pytorch_geometric

# ***************************************************************************
## install fedml from source
# ***************************************************************************
RUN sudo apt-get install -y python3-mpi4py
RUN pip install urllib3==1.26.9
RUN pip3 install --upgrade requests

#
#RUN pip install numpy>=1.21 \
#    PyYAML \
#    h5py \
#    tqdm \
#    wandb \
#    wget \
#    torchvision \
#    paho-mqtt \
#    joblib \
#    boto3 \
#    pynvml \
#    sklearn \
#    networkx \
#    click \
#    grpcio \
#    aif360 \
#    tempeh \
#    imblearn \
#    tabulate

#RUN cd ${INSTALL_DIR} && \
#git clone https://github.com/FedML-AI/FedML.git && cd ${INSTALL_DIR}/FedML && git pull
#RUN cd ${INSTALL_DIR}/FedML/python && \
#git submodule sync && \
#git submodule update --init --recursive --jobs 0 && \
#sudo python setup.py install

COPY ../python
RUN rm -rf ${INSTALL_DIR}/FedML

RUN python -c "import fedml; fedml.__version__"

# ***************************************************************************
## Add fedml user
# ***************************************************************************
# Add a fedml user with user id
RUN useradd --create-home --uid 1000 --shell /bin/bash fedml
RUN usermod -aG sudo fedml
RUN echo "fedml ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Change to non-root privilege
#USER fedml

# Extra installation
#RUN sudo pip3 install sentencepiece
#RUN sudo pip3 install pytorch-ignite
#RUN sudo pip3 install pytest-cov

# Batch Multi Node
ENV USER fedml
ENV HOME /home/$USER
RUN echo $HOME
RUN sudo pip install supervisor


# ***************************************************************************
# SSH Setup
# ***************************************************************************
ENV SSHDIR $HOME/.ssh
RUN sudo mkdir -p ${SSHDIR}
RUN sudo touch ${SSHDIR}/sshd_config
RUN sudo ssh-keygen -t rsa -f ${SSHDIR}/ssh_host_rsa_key -N ''
RUN sudo cp ${SSHDIR}/ssh_host_rsa_key.pub ${SSHDIR}/authorized_keys
RUN sudo cp ${SSHDIR}/ssh_host_rsa_key ${SSHDIR}/id_rsa
RUN sudo chown -R ${USER}:${USER} ${SSHDIR}/
RUN sudo echo "       IdentityFile ${SSHDIR}/id_rsa" >> ${SSHDIR}/config \
&& sudo echo "       StrictHostKeyChecking no" >> ${SSHDIR}/config \
&& sudo echo "       UserKnownHostsFile /dev/null" >> ${SSHDIR}/config \
&& sudo echo "       Port 2022" >> ${SSHDIR}/config \
&& sudo echo 'Port 2022' >> ${SSHDIR}/sshd_config \
&& sudo echo 'UsePrivilegeSeparation no' >> ${SSHDIR}/sshd_config \
&& sudo echo "HostKey ${SSHDIR}/ssh_host_rsa_key" >> ${SSHDIR}/sshd_config
RUN sudo echo "PidFile ${SSHDIR}/sshd.pid" >> ${SSHDIR}/sshd_config
RUN sudo cat ${SSHDIR}/sshd_config
RUN sudo cat ${SSHDIR}/config

RUN sudo chmod -R 600 ${SSHDIR}/*
RUN sudo chown -R ${USER}:${USER} ${SSHDIR}/
RUN eval `ssh-agent -s` && ssh-add ${SSHDIR}/id_rsa

RUN sudo apt install -y iproute2

EXPOSE 22

##############################################################################
# Supervisor container startup
##############################################################################
ADD ./x86-64/supervisord.conf /etc/supervisor/supervisord.conf
ADD ./x86-64/sync_all_nodes.sh /supervised-scripts/sync_all_nodes.sh
RUN sudo chmod 755 supervised-scripts/sync_all_nodes.sh

##############################################################################
# Entry Point Script
##############################################################################
ADD ./x86-64/entry-point.sh /batch-runtime-scripts/entry-point.sh
RUN sudo chmod 0755 /batch-runtime-scripts/entry-point.sh
CMD /batch-runtime-scripts/entry-point.sh