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

if [ "$1" = "--debug" ]; then
  make_or_clean_dir build_x86_linux_debug && cd build_x86_linux_debug
  echo "debug mode"
  cmake .. -DMNN_BUILD_TRAIN=ON -DDEBUG_MODE=ON || exit 1;
  make -j16 || exit 1;
else
  make_or_clean_dir build_x86_linux && cd build_x86_linux
  cmake .. -DMNN_BUILD_TRAIN=ON || exit 1;
  make -j16 || exit 1;
fi

