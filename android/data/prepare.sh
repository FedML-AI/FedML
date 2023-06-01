#!/bin/bash

if [ "$1" = "0" ]
then
   MNIST_DIR=client0/mnist/
elif [ "$1" = "1" ]
then
   MNIST_DIR=client0/mnist/
elif [ "$1" = "2" ]
then
   MNIST_DIR=client0/mnist/
elif [ "$1" = "3" ]
then
   MNIST_DIR=client0/mnist/
else
   MNIST_DIR=client/mnist/
fi
ANDROID_DIR=/storage/emulated/0/Android/data/ai.fedml.edgedemo/files/dataset

adb push $MNIST_DIR $ANDROID_DIR
#rm -rf mnist


#adb push $CIFAR10_DIR $ANDROID_DIR
