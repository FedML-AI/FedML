#!/bin/bash

set -e
set -x

if [ "$GITHUB_EVENT_NAME" == "schedule" ]; then
    ANACONDA_ORG="fedml-wheels-nightly"
    ANACONDA_TOKEN="$FEDML_NIGHTLY_UPLOAD_TOKEN"
else
    ANACONDA_ORG="fedml-wheels-staging"
    ANACONDA_TOKEN="$FEDML_STAGING_UPLOAD_TOKEN"
fi

export PATH=$CONDA/bin:$PATH
conda create -n upload -y python=3.8
source activate upload
conda install -y fedml

# Force a replacement if the remote file already exists
anaconda -t $ANACONDA_TOKEN upload --force -u $ANACONDA_ORG dist/artifact/*
echo "Index: https://pypi.anaconda.org/$ANACONDA_ORG/simple"