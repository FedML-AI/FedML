FROM nvidia/cuda:11.0.3-devel-ubuntu18.04@sha256:e0db1c5ab7ef25027f710bfbf7b2cf1fa0588888e952954f888d2828583e2689


ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get clean
RUN apt-get --allow-downgrades update
RUN apt-get -f install -y --no-install-recommends \
        software-properties-common build-essential autotools-dev \
        nfs-common pdsh \
        cmake g++ gcc \
        curl wget vim tmux emacs less unzip \
        htop iftop iotop ca-certificates openssh-client openssh-server \
        rsync iputils-ping net-tools sudo \
        llvm-9-dev

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
RUN apt-get --allow-downgrades update
RUN apt-get install -y python3 python3-pip
RUN ln -nsf /usr/bin/python3 /usr/bin/python
RUN ln -nsf /usr/bin/pip3 /usr/bin/pip

# https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-11-now-available/72048?page=52
#RUN wget https://nvidia.box.com/shared/static/ssf2v7pf5i245fk4i0q926hy4imzs2ph.whl -O torch-1.11.0-cp38-cp38-linux_aarch64.whl
#RUN sudo apt-get  -y --no-install-recommends install python3-pip libopenblas-base libopenmpi-dev libomp-dev
#RUN pip install numpy torch-1.11.0-cp38-cp38-linux_aarch64.whl

# ***************************************************************************
# Git
# ***************************************************************************
RUN add-apt-repository ppa:git-core/ppa -y && \
    apt-get --allow-downgrades update && \
    apt-get install -y git && \
    git --version

# ***************************************************************************
## install fedml from source
# ***************************************************************************
RUN sudo apt-get install -y python3-mpi4py
RUN pip install urllib3==1.26.9
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
#    matplotlib \
#    grpcio \
#    aif360 \
#    tempeh \
#    imblearn \
#    tabulate

RUN cd ${INSTALL_DIR} && \
git clone https://github.com/FedML-AI/FedML.git

RUN cd ${INSTALL_DIR}/FedML/python && \
git submodule sync && \
git submodule update --init --recursive --jobs 0 && \
sudo python setup.py install
#RUN rm -rf ${INSTALL_DIR}/FedML

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