#!/usr/bin/env bash

DATA_PATH=~/.cache/fedml_data/stackoverflow
mkdir $DATA_PATH
cd $DATA_PATH
wget --no-check-certificate --no-proxy https://fedml.s3-us-west-1.amazonaws.com/stackoverflow.tag_count.tar.bz2
wget --no-check-certificate --no-proxy https://fedml.s3-us-west-1.amazonaws.com/stackoverflow.word_count.tar.bz2
wget --no-check-certificate --no-proxy https://fedml.s3-us-west-1.amazonaws.com/stackoverflow.tar.bz2
wget --no-check-certificate --no-proxy https://fedml.s3-us-west-1.amazonaws.com/stackoverflow_nwp.pkl

tar -xvf stackoverflow.tag_count.tar.bz2
tar -xvf stackoverflow.word_count.tar.bz2
tar -xvf stackoverflow.tar.bz2
