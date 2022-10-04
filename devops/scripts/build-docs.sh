#!/usr/bin/env bash

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $WORKSPACE/miniconda
hash -r
source $WORKSPACE/miniconda/etc/profile.d/conda.sh
conda init bash
conda config --set always_yes yes --set changeps1 no
conda update -q conda

# Useful for debugging any issues with conda
conda info -a
conda config --add channels defaults
conda config --add channels conda-forge
conda config --add channels bioconda

conda create -y -n fedml-docs python=3.8
conda activate fedml-docs
conda install sphinx
conda install -c conda-forge myst-parser

cd doc/en/
make html
make clean html
cd ../../