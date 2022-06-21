#!/bin/bash -l

#set -e
#set -x

if [[ "$RUNNER_OS" == "windows-*" ]]; then
    conda install -c conda-forge mpi4py openmpi
fi

python -m pip install cibuildwheel
python -m cibuildwheel --output-dir wheelhouse