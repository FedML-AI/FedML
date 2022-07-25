#!/bin/bash

echo "nameserver 8.8.8.8" > /etc/resolv.conf

which conda
conda info
conda install sphinx
conda install -c conda-forge myst-parser
cd doc/en/
make html
make clean html
cd ../../