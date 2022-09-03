#!/usr/bin/env bash

BUILD_ROOT=`pwd`

function make_or_clean_dir {
  if [ -d $1 ]; then
#    rm -rf $1/*
    echo "incremental compilation"
  else
    mkdir $1
  fi
}

make_or_clean_dir build_x86_linux && cd build_x86_linux
cmake .. -DMNN_BUILD_TRAIN=ON || exit 1;
make -j16 || exit 1;

