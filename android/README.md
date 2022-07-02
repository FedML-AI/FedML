# Android and Mobile NN Code Architecture

- Android project root path: https://github.com/FedML-AI/FedML/tree/master/android

The architecture is divided into three vertical layers and multiple horizontal modules:

### 1. Android APK Layer
- app

https://github.com/FedML-AI/FedML/tree/master/android/app


- fedmlsdk_demo

https://github.com/FedML-AI/FedML/tree/master/android/fedmlsdk_demo

### 2. Android SDK layer (Java API + JNI + So library)

https://github.com/FedML-AI/FedMLAndroidSDK


### 3. MobileNN: FedML Mobile Training Engine Layer (C++, MNN, PyTorch, etc.)

https://github.com/FedML-AI/MobileNN

https://github.com/FedML-AI/MNN

https://github.com/FedML-AI/pytorch

At this stage, the app layer is open sourced, the Android SDK is released to the open source community, and the Mobile NN C++ layer is close source.

## Tutorial
https://doc.fedml.ai/cross-device/examples/mqtt_s3_fedavg_mnist_lr_example.html

## About Authors

FedML team has more than 5 years experience in industrial grade Android development. See CTO and Senior Android Engineer's project experience in Android at https://chaoyanghe.com/industrial-experience/