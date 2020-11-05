#!/bin/bash
# set -ex


# install pyflakes to do code error checking
echo "pip3 install pyflakes --cache-dir $HOME/.pip-cache"
pip3 install pyflakes --cache-dir $HOME/.pip-cache

# Conda Installation
wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-armv7l.sh
bash Miniconda3-latest-Linux-armv7l.sh -b -p $HOME/miniconda
echo 'export PATH=$PATH:/home/pi/miniconda/bin' >> ~/.bashrc
source ~/.bashrc

conda install anaconda-client
anaconda search -t conda blas
anaconda search -t conda openblas
conda config --add channels rpi

echo "conda create -n fedml python=3.7.4"
conda create -n fedml python=3.7.4

echo "conda activate fedml"
conda activate fedml

# Install PyTorch 1.7: https://mathinf.com/pytorch/arm64/
sudo apt-get install python3-numpy python3-wheel python3-setuptools python3-future python3-yaml python3-six python3-requests python3-pip python3-pillow
sudo pip3 install torch*.whl torchvision*.whl

# Install MPI
conda install -c anaconda mpi4py

# Install Wandb
pip install --upgrade wandb

# Install other required package
conda install scikit-learn
conda install numpy
conda install h5py
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

# 2. FederatedEMNIST
cd ./data/FederatedEMNIST
sh download_federatedEMNIST.sh
cd ../../

# 3. shakespeare
cd ./data/shakespeare
sh download_shakespeare.sh
cd ../../


# 4. fed_shakespeare
cd ./data/fed_shakespeare
sh download_shakespeare.sh
cd ../../

# 5. fed_cifar100
cd ./data/fed_cifar100
sh download_fedcifar100.sh
cd ../../

# 6. stackoverflow
cd ./data/stackoverflow
sh download_stackoverflow.sh
cd ../../

# 7. CIFAR10
cd ./data/cifar10
sh download_cifar10.sh
cd ../../

# 8. CIFAR100
cd ./data/cifar100
sh download_cifar100.sh
cd ../../

# 9. CINIC10
cd ./data/cinic10
sh download_cinic10.sh > cinic10_downloading_log.txt
cd ../../
