#!/usr/bin/env bash

#wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
#bash miniconda.sh -b -p $WORKSPACE/miniconda
#hash -r
#conda config --set always_yes yes --set changeps1 no
#conda update -q conda
#
## Useful for debugging any issues with conda
#conda info -a
#conda config --add channels defaults
#conda config --add channels conda-forge
#conda config --add channels bioconda
#
#/bin/bash
#source /etc/profile
#source ~/.bashrc
#source $WORKSPACE/miniconda/etc/profile.d/conda.sh
#
#which conda
#conda info
#conda install sphinx
#conda install -c conda-forge myst-parser

pip install --upgrade pip
pip install -U sphinx
pip install myst-parser

cd doc/en/
make html
make clean html
cd ../../