#!/bin/bash
set -ex

# install pyflakes to do code error checking
echo "pip3 install pyflakes --cache-dir $HOME/.pip-cache"
pip3 install pyflakes --cache-dir $HOME/.pip-cache

# Conda Installation
wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
echo "you have managed"
bash miniconda.sh -b -p $HOME/miniconda
source "$HOME/miniconda/etc/profile.d/conda.sh"
echo "you have managed2"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a

echo "conda create -n fedml python=3.7.4"
conda create -n fedml python=3.7.4

echo "conda activate fedml"
conda activate fedml

# Install PyTorch (please visit pytorch.org to check your version according to your physical machines
conda install -c anaconda cudatoolkit

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
pip install -r requirements.txt

## install the dataset
## 1. MNIST
#cd ./data/MNIST
#sh download_and_unzip.sh
#cd ../../
#
## 2. FederatedEMNIST
#cd ./data/FederatedEMNIST
#sh download_federatedEMNIST.sh
#cd ../../
#
## 3. shakespeare
#cd ./data/shakespeare
#sh download_shakespeare.sh
#cd ../../
#
#
## 4. fed_shakespeare
#cd ./data/fed_shakespeare
#sh download_shakespeare.sh
#cd ../../
#
## 5. fed_cifar100
#cd ./data/fed_cifar100
#sh download_fedcifar100.sh
#cd ../../
#
## 6. stackoverflow
## cd ./data/stackoverflow
## sh download_stackoverflow.sh
## cd ../../
#
## 7. CIFAR10
#cd ./data/cifar10
#sh download_cifar10.sh
#cd ../../
#
## 8. CIFAR100
#cd ./data/cifar100
#sh download_cifar100.sh
#cd ../../
#
## 9. CINIC10
#cd ./data/cinic10
#sh download_cinic10.sh > cinic10_downloading_log.txt
#cd ../../

## 10. Downlaod Synthea Taken from the Pyvertical Example ToDo Adjust it!
##!/bin/bash
#
## # Download Synthea Data
#
## ----------------------------------------------
## If you choose to download Synthea on the PyVertical/data/synthea folder, execute:
#
## cd ../data/
## GITKEEP_exists=/synthea/.gitkeep
## if test -f "$GITKEEP_exists"; then
##     rm /synthea/.gitkeep
## fi
#
## git clone https://github.com/synthetichealth/synthea.git
#
## ----------------------------------------------
#
#
## Generate data
#cd 'PATH'/synthea # Change 'PATH' to the correct path on your system
#./run_synthea -s 42 -p 5000 --exporter.csv.export true
#
#
## Copy data to PyVertical/data
#mv output/csv/*csv ../../data/
#
## Remove unnecessary files
#cd ../../data/
#
#rm allergies.csv careplans.csv devices.csv encounters.csv imaging_studies.csv
#rm immunizations.csv organizations.csv payers.csv payer_transitions.csv
#rm procedures.csv providers.csv supplies.csv
