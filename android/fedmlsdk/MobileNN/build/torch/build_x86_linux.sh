#!/usr/bin/env bash

# build pytorch
if [ ! -d "./../../pytorch/build_mobile" ]; then
bash ./../../pytorch/scripts/build_mobile.sh || exit 1;
fi

function make_or_clean_dir {
  if [ -d $1 ]; then
#    rm -rf $1/*
    echo "incremental compilation"
  else
    mkdir $1
  fi
}

# build our source code
make_or_clean_dir build_x86_linux && cd build_x86_linux
cmake .. || exit 1;
make -j16 || exit 1;