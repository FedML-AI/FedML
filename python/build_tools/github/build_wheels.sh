#!/bin/bash -l

set -e
set -x

if [[ "$RUNNER_OS" == "windows-*" ]]; then
    conda install -c conda-forge mpi4py openmpi
fi

if [[ "$RUNNER_OS" == "macOS" ]]; then
    if [[ "$CIBW_BUILD" == *-macosx_arm64 ]]; then
        export PYTHON_CROSSENV=1
        export MACOSX_DEPLOYMENT_TARGET=12.0
        OPENMP_URL="https://anaconda.org/conda-forge/llvm-openmp/11.1.0/download/osx-arm64/llvm-openmp-11.1.0-hf3c4609_1.tar.bz2"
    else
        export MACOSX_DEPLOYMENT_TARGET=10.9
        OPENMP_URL="https://anaconda.org/conda-forge/llvm-openmp/11.1.0/download/osx-64/llvm-openmp-11.1.0-hda6cdc1_1.tar.bz2"
    fi

    sudo conda create -n build $OPENMP_URL
    PREFIX="/usr/local/miniconda/envs/build"

    export CC=/usr/bin/clang
    export CXX=/usr/bin/clang++
    export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
    export CFLAGS="$CFLAGS -I$PREFIX/include"
    export CXXFLAGS="$CXXFLAGS -I$PREFIX/include"
    export LDFLAGS="$LDFLAGS -Wl,-rpath,$PREFIX/lib -L$PREFIX/lib -lomp"
fi

python -m pip install cibuildwheel
python -m cibuildwheel --output-dir wheelhouse