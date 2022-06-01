#!/bin/bash

is_building_gpu_image=$1

echo "conda set ssl_verify"

conda clean -i

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/

conda config --set show_channel_urls yes

conda config --set ssl_verify false

conda update conda

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
