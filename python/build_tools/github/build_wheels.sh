#!/bin/bash

set -e
set -x

python -m pip install cibuildwheel==2.5.0
python -m cibuildwheel --output-dir wheelhouse