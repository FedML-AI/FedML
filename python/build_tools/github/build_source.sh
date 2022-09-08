#!/bin/bash

set -e
set -x

# Move up two levels to create the virtual
# environment outside of the source folder
cd ../../
pwd
ls

python -m venv build_env
source build_env/bin/activate

python -m pip install -U wheel setuptools
python -m pip install numpy scipy cython
python -m pip install twine

cd FedML/python

python setup.py sdist bdist_wheel

# Check whether the source distribution will render correctly
twine check dist/*.tar.gz