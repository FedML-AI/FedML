#!/usr/bin/env bash

# use ndk 21.x
export ANDROID_NDK=/Users/leigao/Library/Android/sdk/ndk/21.4.7075529
export ANDROID_ABI=arm64-v8a
export BUILD_LITE_INTERPRETER=0

# build pytorch
if [ ! -d "./../../pytorch/build_android" ]; then
bash ./../../pytorch/scripts/build_android.sh || exit 1;
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
make_or_clean_dir build_arm_android_64 && cd build_arm_android_64
cmake ..  \
      -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DANDROID_ABI="arm64-v8a" \
      -DANDROID_STL=c++_static \
      -DANDROID_NATIVE_API_LEVEL=android-21 \
      -DANDROID_TOOLCHAIN=clang \
      -DBUILD_ANDROID=ON || exit 1;
make -j16 || exit 1;

