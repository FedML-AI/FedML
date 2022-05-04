FROM nvidia/cuda:11.1-devel-ubuntu18.04


##############################################################################
# Directory Settings
##############################################################################
ENV INSTALL_DIR=/tmp
ENV WORKSPACE=/home/fedml
RUN mkdir -p ${INSTALL_DIR}
RUN mkdir -p ${WORKSPACE}


##############################################################################
# Installation/Basic Utilities
##############################################################################
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
        software-properties-common build-essential autotools-dev \
        nfs-common pdsh \
        cmake g++ gcc \
        curl wget vim tmux emacs less unzip \
        htop iftop iotop ca-certificates openssh-client openssh-server \
        rsync iputils-ping net-tools sudo \
        llvm-9-dev

##############################################################################
# Installation Latest Git
##############################################################################
RUN add-apt-repository ppa:git-core/ppa -y && \
    apt-get update && \
    apt-get install -y git && \
    git --version

##############################################################################
# OPENMPI
##############################################################################
ENV OPENMPI_BASEVERSION=4.0
ENV OPENMPI_VERSION=${OPENMPI_BASEVERSION}.1
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

##############################################################################
# Python
##############################################################################
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3
RUN apt-get install -y python3 python3-dev && \
    rm -f /usr/bin/python && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    curl -O https://bootstrap.pypa.io/get-pip.py && \
        python get-pip.py && \
        rm get-pip.py && \
    pip install --upgrade pip && \
    # Print python an pip version
    python -V && pip -V
RUN sudo sudo -H pip3 install --ignore-installed PyYAML
RUN pip install ipython

RUN apt-get update && \
    apt-get install -y vim git tmux wget curl autoconf libtool apt-utils

##############################################################################
# NCCL 2.10.3 Upgrade
##############################################################################
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub && add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /" && apt update && apt install -y --allow-change-held-packages libnccl2=2.10.3-1+cuda11.0 libnccl-dev=2.10.3-1+cuda11.0
ENV NCCL_VERSION=2.10.3

##############################################################################
# MPI and other libraries
##############################################################################
RUN pip3 install mpi4py \
    grpcio \
    scikit-learn \
    numpy \
    h5py \
    setproctitle \
    networkx \
    requests \
    tqdm \
    scipy \
    scikit-learn \
    seqeval \
    tensorboardx \
    pandas \
    wandb \
    streamlit \
    matplotlib \
    seaborn \
    certifi==2020.6.20 \
    chardet==3.0.4 \
    et-xmlfile==1.0.1 \
    idna==2.10 \
    jdcal==1.4.1 \
    ruamel.yaml==0.16.10 \
    ruamel.yaml.clib==0.2.0 \
    urllib3==1.25.9 \
    flask==1.1.2 \
    gunicorn==20.0.4 \
    gevent==20.6.2 \
    paho-mqtt==1.5.0 \
    Celery==4.4.7 \
    dill==0.3.3 \
    aif360 \
    tempeh \
    imblearn \
    tabulate


##############################################################################
# PyTorch (latest version: v1.9.0)
##############################################################################
RUN sudo pip3 install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses h5py
RUN sudo pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html


##############################################################################
# install torch-geometric (https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
##############################################################################
RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
RUN pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
RUN pip install torch-geometric


##############################################################################
# pip install FedML
##############################################################################
RUN pip3 install --upgrade fedml


##############################################################################
## Add fedml user
###############################################################################
# Add a fedml user with user id 8877
RUN useradd --create-home --uid 1000 --shell /bin/bash fedml
RUN usermod -aG sudo fedml
RUN echo "fedml ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
# Change to non-root privilege
USER fedml

# Extra installation
RUN sudo pip3 install sentencepiece
RUN sudo pip3 install pytorch-ignite
RUN sudo pip3 install pytest-cov

# Batch Multi Node
ENV USER fedml
ENV HOME /home/$USER
RUN echo $HOME
RUN sudo pip install supervisor


##############################################################################
# SSH Setup
##############################################################################
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


USER fedml

##############################################################################
# Supervisor container startup
##############################################################################
ADD supervisord.conf /etc/supervisor/supervisord.conf
ADD sync_all_nodes.sh /supervised-scripts/sync_all_nodes.sh
RUN sudo chmod 755 supervised-scripts/sync_all_nodes.sh

##############################################################################
# Entry Point Script
##############################################################################
ADD entry-point.sh /batch-runtime-scripts/entry-point.sh
RUN sudo chmod 0755 /batch-runtime-scripts/entry-point.sh
CMD /batch-runtime-scripts/entry-point.sh