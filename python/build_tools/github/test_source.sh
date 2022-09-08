#!/bin/bash

set -e
set -x

cd ../../

python -m venv test_env
source test_env/bin/activate

python -m pip install -U wheel setuptools
python -m pip install FedML/python/dist/*.tar.gz
python -c "import fedml; fedml.__version__"