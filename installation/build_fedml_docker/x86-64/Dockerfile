
# https://hub.docker.com/r/nvidia/cuda/tags
# https://hub.docker.com/r/nvidia/cuda/tags?page=1&name=11.6.0-devel-ubuntu was released at May 6th, 2022
ARG NVIDIA_BASE_IMAGE=nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
FROM ${NVIDIA_BASE_IMAGE}
ARG PYTORCH_VERSION=2.1.0

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN export CUDA_HOME=/usr/local/cuda
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
RUN export PATH=$PATH:$CUDA_HOME/bin


# ***************************************************************************
# Version and directory Settings
# ***************************************************************************
ENV INSTALL_DIR=/tmp
ENV WORKSPACE=/home/fedml
RUN mkdir -p ${INSTALL_DIR}
RUN mkdir -p ${WORKSPACE}


# ***************************************************************************
# Utilities
# ***************************************************************************
RUN apt update -y && apt install -y --no-install-recommends \
        software-properties-common build-essential autotools-dev \
        nfs-common pdsh \
        cmake g++ gcc \
        curl wget vim tmux emacs less unzip \
        htop iftop iotop ca-certificates openssh-client openssh-server \
        rsync iputils-ping net-tools sudo \
        python3-pip git git-lfs

RUN git lfs install


# ***************************************************************************
## install fedml and pytorch
# ***************************************************************************

RUN pip install -U fedml torch==${PYTORCH_VERSION} --no-cache-dir


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
#RUN sudo pip install supervisor
RUN cd ${INSTALL_DIR}


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