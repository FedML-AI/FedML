#!/usr/bin/env bash

# ANDROID_NDK=/Users/leigao/Library/Android/sdk/ndk/24.0.8215888
ANDROID_NDK=/Users/chaoyanghe/Library/Android/sdk/ndk/24.0.8215888

BUILD_ROOT=`pwd`

function make_or_clean_dir {
  if [ -d $1 ]; then
    rm -rf $1/*
  else
    mkdir $1
  fi
}

make_or_clean_dir build_arm_android_64 && cd build_arm_android_64
cmake ..  \
      -DMOBILE_BACKEND=MNN \
      -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DANDROID_ABI="arm64-v8a" \
      -DANDROID_STL=c++_static \
      -DCMAKE_BUILD_TYPE=Release \
      -DANDROID_NATIVE_API_LEVEL=android-32 \
      -DANDROID_TOOLCHAIN=clang \
      -DMNN_BUILD_FOR_ANDROID_COMMAND=true \
      -DMNN_BUILD_TRAIN=ON \
      -DNATIVE_LIBRARY_OUTPUT=. \
      -DNATIVE_INCLUDE_OUTPUT=. || exit 1;
make -j16 || exit 1;
cd $BUILD_ROOT
