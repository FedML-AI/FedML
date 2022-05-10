# https://hub.docker.com/r/nvidia/cuda/tags
# https://hub.docker.com/r/nvidia/cuda/tags?page=1&name=11.6.0-devel-ubuntu was released at May 6th, 2022
FROM nvidia/cuda:11.6.0-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get --allow-downgrades update
RUN apt-get install -y python3 python3-pip
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -nsf /usr/bin/pip3 /usr/bin/pip

RUN apt-get install -y python3-mpi4py

RUN git clone https://github.com/FedML-AI/FedML.git && \
cd ./FedML && git checkout e798061d62560b03e049d514e7cc8f1a753fde6b && \
RUN cd python && \
git submodule sync && \
git submodule update --init --recursive --jobs 0 && \
sudo python setup.py install

RUN python -c "import torch; torch.__version__"
RUN python -c "import fedml"

git clone https://github.com/FedML-AI/FedML.git && \
cd ./FedML && git checkout e798061d62560b03e049d514e7cc8f1a753fde6b && \
cd python && \
python setup.py install