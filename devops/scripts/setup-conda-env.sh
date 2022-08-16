#!/bin/bash

#echo "nameserver 8.8.8.8" > /etc/resolv.conf

echo "conda create -n fedml python=3.7.4"
conda create -y -n fedml-pip python=3.7.4

echo "conda activate fedml"
conda activate fedml-pip

# Install MPI
conda install -y -c anaconda mpi4py

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'
