#!/bin/bash

# Thanks to the scikit-learn team, this script is ported from:
# https://github.com/scikit-learn/scikit-learn/blob/main/build_tools/github/build_minimal_windows_image.sh

set -e
set -x

PYTHON_VERSION=$1
BITNESS=$2

if [[ "$BITNESS" == "32" ]]; then
    # 32-bit architectures are not supported
    # by the official Docker images: Tests will just be run
    # on the host (instead of the minimal Docker container).
    exit 0
fi

TEMP_FOLDER="$HOME/AppData/Local/Temp"
WHEEL_PATH=$(ls -d $TEMP_FOLDER/**/*/repaired_wheel/*)
WHEEL_NAME=$(basename $WHEEL_PATH)

cp $WHEEL_PATH $WHEEL_NAME

# Dot the Python version for identyfing the base Docker image
PYTHON_VERSION=$(echo ${PYTHON_VERSION:0:1}.${PYTHON_VERSION:1:2})

# Build a minimal Windows Docker image for testing the wheels
docker build --build-arg PYTHON_VERSION=$PYTHON_VERSION \
             --build-arg WHEEL_NAME=$WHEEL_NAME \
             --build-arg CONFTEST_NAME=$CONFTEST_NAME \
             --build-arg CIBW_TEST_REQUIRES="$CIBW_TEST_REQUIRES" \
             -f build_tools/github/Windows \
             -t scikit-learn/minimal-windows .