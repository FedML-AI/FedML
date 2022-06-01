#!/bin/bash

is_building_gpu_image=$1

echo "conda set ssl_verify"

cat /root/.condarc

conda config --set ssl_verify false

conda token set --no-ssl-verify `conda install conda-token -n root`

conda info

conda_base_dir=`conda info |grep  'base environment' |awk -F':' '{print $2}' |awk -F'(' '{print $1}' |awk -F' ' '{print $1}'`
conda_env_init="${conda_base_dir}/etc/profile.d/conda.sh"
source ${conda_env_init}

echo "conda create -n fedml python=3.7.4"
conda create -y -n fedml-pip python=3.7.4

echo "conda activate fedml"
conda activate fedml-pip

# Install MPI
conda install -y -c anaconda mpi4py

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'
