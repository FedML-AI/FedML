# Mobile MN 
This demo provides CMake build scripts to compile a source code, `demo.cpp`, that trains a LeNet model in MNN framework, into an executable target file `demo.out` along with MNN shared libraries for Linux/macOS and Android platforms.

## Environmental Requirements
* cmake (version >=3.10 is recommended)
* protobuf (version >= 3.0 is required)
* gcc (version >= 4.9 is required)

## Linux/macOS
1. run `./build_x86_linux.sh`, which will generate the executable file `demo.out` under build_x86_linux folder.

```
cd build/train
sh build_x86_linux.sh
./build_x86_linux/demo.out mnist ../../../data/lenet_mnist.mnn ../../../data/mnist

```

## Android
1. [Download and Install NDK](https://developer.android.com/ndk/downloads/), latest release version is recommended
2. Set ANDROID_NDK path at line 3 in `build_arm_android_64.sh`, eg: ANDROID_NDK=/Users/username/path/to/Android-ndk-r14b
3. run `./build_arm_android_64.sh`, which will generate the executable file `demo.out` under build_arm_android_64 folder.
4. run `./test_arm_android_64.sh`, which will push `demo.out` to your android device and execute it.

## Notes
1. You can change CMake compilation options in `build.sh` as needed (i.e. turn off demo/quantools/evaluation/converter/test/benchmark options, turn on openmp/opencl/opengl/vulkan as your backend, set FP16/BF16 low precision mode, and etc). Check [MNN document](https://www.yuque.com/mnn/en/cmake_opts) and [MNN CMakeList](https://github.com/alibaba/MNN/blob/master/CMakeLists.txt) for more information.
2. MNN compilation artifacts under `build/mnn_binary_dir`
   * libMNN: Backend Shared Library 
   * libMNNTrain: Training Framework Shared Library 
   * libMNNExpr: Express Training API Shared Library
3. To run `demo.out` on your linux/macOS machine, first download MNIST dataset from [Google Drive](https://drive.google.com/drive/folders/1IB1-NJgzHSEb7ucgJzM2Gj8QzxpYAjGy?usp=sharing), and run `./demo.out /path/to/data/mnist_data`. 
To run `demo.out` on your android device, adb push mnist_data to your android device under `/data/local/tmp` before running `./test_arm_android_64.sh`. 

## Dependency
MNN: 
https://github.com/FedML-AI/MNN.git

pytorch: 
https://github.com/FedML-AI/pytorch.git