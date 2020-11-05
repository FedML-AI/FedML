#!/bin/bash


# Conda Installation
wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-armv7l.sh
bash Miniconda3-latest-Linux-armv7l.sh -b -p $HOME/miniconda
echo 'export PATH=$PATH:/home/pi/miniconda/bin' >> ~/.bashrc
source ~/.bashrc

conda install anaconda-client
anaconda search -t conda blas
anaconda search -t conda openblas
anaconda search -t conda python
conda config --add channels rpi

echo "conda create -n fedml"
conda create -n fedml

echo "source activate fedml"
source activate fedml

alias python='/usr/bin/python3.7'
alias pip=pip3

# Install PyTorch 1.7: https://mathinf.com/pytorch/arm64/
sudo apt install libopenblas-base libopenblas-dev libblas-dev m4 cmake cython python3-dev python3-yaml python3-setuptools python3-wheel python3-pillow python3-numpy
sudo pip3 install ./pytorch-pkg-on-rpi/torch-1.4.0a0+7f73f1d-cp37-cp37m-linux_armv7l.whl
sudo pip3 install ./pytorch-pkg-on-rpi/torchvision-0.5.0a0+85b8fbf-cp37-cp37m-linux_armv7l.whl
cd /usr/local/lib/python3.7/dist-packages/torch
sudo mv _C.cpython-37m-arm-linux-gnueabi.so _C.so
sudo mv _dl.cpython-37m-arm-linux-gnueabi.so _dl.so

# install again, to make sure pytorch related packages are installed
sudo apt install libopenblas-base libopenblas-dev libblas-dev m4 cmake cython python3-dev python3-yaml python3-setuptools python3-wheel python3-pillow python3-numpy

# Install Wandb
pip3 install --upgrade wandb

# Install other required package
conda install scikit-learn
conda install numpy
conda install h5py
conda install setproctitle
conda install networkx
pip3 install requests

pip3 install -r requirements.txt

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
