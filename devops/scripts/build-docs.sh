#!/usr/bin/env bash

/bin/bash
source /etc/profile
source ~/.bashrc
source $WORKSPACE/miniconda/etc/profile.d/conda.sh

which conda
conda info
conda install sphinx
conda install -c conda-forge myst-parser
cd doc/en/
make html
make clean html
cd ../../