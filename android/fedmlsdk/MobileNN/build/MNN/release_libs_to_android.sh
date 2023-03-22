#!/usr/bin/env bash

cp -rf ./build_arm_android_64/mnn_binary_dir/libMNN.so ../../../libs/MNN/arm64-v8a
cp -rf ./build_arm_android_64/mnn_binary_dir/libMNN_Express.so ../../../libs/MNN/arm64-v8a
cp -rf ./build_arm_android_64/mnn_binary_dir/tools/train/libMNNTrain.so ../../../libs/MNN/arm64-v8a