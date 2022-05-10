# https://hub.docker.com/r/nvidia/cuda/tags
# https://hub.docker.com/r/nvidia/cuda/tags?page=1&name=11.6.0-devel-ubuntu was released at May 6th, 2022
FROM nvidia/cuda:11.6.0-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get --allow-downgrades update
RUN apt-get install -y python3 python3-pip
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -nsf /usr/bin/pip3 /usr/bin/pip

RUN apt-get install -y python3-mpi4py
RUN pip install fedml

RUN python -c "import torch; torch.__version__"
RUN python -c "import fedml"


# ***************************************************************************
## Add fedml user
# ***************************************************************************
# Add a fedml user with user id
RUN useradd --create-home --uid 1000 --shell /bin/bash fedml
RUN usermod -aG sudo fedml
RUN echo "fedml ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
