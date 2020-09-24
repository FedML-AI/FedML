#!/bin/bash
set -ex

# install pyflakes to do code error checking
echo "pip3 install pyflakes --cache-dir $HOME/.pip-cache"
pip3 install pyflakes --cache-dir $HOME/.pip-cache

# Conda Installation
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
source "$HOME/miniconda/etc/profile.d/conda.sh"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a

echo "conda create -n fedml python=3.7.4"
conda create -n fedml python=3.7.4

echo "conda activate fedml"
conda activate fedml

# Install PyTorch (please visit pytorch.org to check your version according to your physical machines
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# Install MPI
conda install -c anaconda mpi4py

# Install Wandb
pip install --upgrade wandb

# Install other required package
conda install scikit-learn
conda install numpy
conda install scipy
conda install setproctitle
conda install networkx

cd ./fedml_mobile/server/executor
pip install -r requirements.txt
cd ./../../../

# install the dataset
# 1. MNIST
cd ./data/MNIST
sh download_and_unzip.sh
cd ../../

# 2. CIFAR10
cd ./data/cifar10
sh download_cifar10.sh
cd ../../
