#!/bin/bash

is_building_gpu_image=$1

echo "conda create -n fedml python=3.7.4"
conda create -y -n fedml-pip python=3.7.4

echo "conda activate fedml"
conda activate fedml-pip

# Install MPI
conda install -y -c anaconda mpi4py

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'
