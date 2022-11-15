FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]
RUN sed -i ~/.profile -e 's/mesg n || true/tty -s \&\& mesg n/g'
RUN apt update -y && apt upgrade -y
RUN apt-get update && apt-get install build-essential -y
RUN apt-get install git -y
RUN apt-get install cmake -y
RUN apt-get install autoconf -y
RUN apt-get install -y clang libomp5 libomp-dev
RUN apt-get install -y ninja-build libmpfr-dev libgmp-dev libboost-all-dev
RUN apt install vim -y
RUN apt-get update && apt-get install -y software-properties-common gcc && \
    add-apt-repository -y ppa:deadsnakes/ppa

RUN apt-get update && apt-get install -y python3.9 python3-distutils python3-pip python3-apt
RUN pip3 install pybind11

RUN pip3 install fedml

WORKDIR /root/FedML/python/fedml/core/fhe/fhe-fed
RUN git clone https://github.com/weidai11/cryptopp.git
RUN cd cryptopp && make && make test && make install

WORKDIR /root/FedML/python/fedml/core/fhe/fhe-fed
RUN git clone -b release-v1.11.2 https://gitlab.com/palisade/palisade-development.git
RUN mkdir -p /root/fhe-fedml/palisade-development/build
WORKDIR /root/FedML/python/fedml/core/fhe/fhe-fed/palisade-development/build
RUN cmake .. && make && make install

COPY . /root/FedML
WORKDIR /root/FedML/python/fedml/core/fhe/fhe-fed/palisade_pybind/SHELFI_FHE/src
RUN pip3 install ../
WORKDIR /root/FedML



# RUN pip3 install matplotlib pandas pytorch_lightning torch torchvision pytorch-tabnet
# RUN pip3 install -U scikit-learn
