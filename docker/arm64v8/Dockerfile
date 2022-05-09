FROM arm64v8/ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get --allow-downgrades update
RUN apt-get install -y --no-install-recommends \
        software-properties-common build-essential autotools-dev \
        nfs-common pdsh \
        cmake g++ gcc \
        curl wget vim tmux emacs less unzip \
        htop iftop iotop ca-certificates openssh-client openssh-server \
        rsync iputils-ping net-tools sudo \
        llvm-9-dev



#RUN sudo apt-get install python3-pip libopenblas-base libopenmpi-dev libomp-dev
RUN sudo apt-get -y --no-install-recommends install python3-pip python3-dev
RUN pip3 install Cython
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN python --version

# https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-11-now-available/72048?page=52
RUN wget https://nvidia.box.com/shared/static/ssf2v7pf5i245fk4i0q926hy4imzs2ph.whl -O torch-1.11.0-cp38-cp38-linux_aarch64.whl
RUN sudo apt-get  -y --no-install-recommends install python3-pip libopenblas-base libopenmpi-dev libomp-dev
RUN pip3 install numpy torch-1.11.0-cp38-cp38-linux_aarch64.whl

# ***************************************************************************
# Git
# ***************************************************************************
RUN add-apt-repository ppa:git-core/ppa -y && \
    apt-get --allow-downgrades update && \
    apt-get install -y git && \
    git --version

RUN sudo apt-get install -y python3-mpi4py

RUN pip install --upgrade urllib3

RUN cd ${INSTALL_DIR} && \
git clone https://github.com/FedML-AI/FedML.git && \
cd FedML/python && \
git submodule sync && \
git submodule update --init --recursive --jobs 0 && \
sudo python setup.py install
