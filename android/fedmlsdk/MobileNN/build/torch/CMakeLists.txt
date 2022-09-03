cmake_minimum_required(VERSION 3.0)

option(BUILD_ANDROID "Build for Android" OFF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

set(TARGET "main_torch_train.out")

# path to  MobileNN directory
set(MOBILENN_HOME_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../MobileNN")

# enable TORCH backend
add_definitions(-DUSE_TORCH_BACKEND)

# path to pytorch directory
set(PYTORCH_HOME_DIR "${MOBILENN_HOME_DIR}/pytorch")

# Source code path
file(GLOB_RECURSE DEMO_SRC
        ${MOBILENN_HOME_DIR}/src/torch/*.cpp
        ${MOBILENN_HOME_DIR}/src/train/FedMLBaseTrainer.cpp
        ${MOBILENN_HOME_DIR}/src/train/FedMLTorchTrainer.cpp
        ${MOBILENN_HOME_DIR}/src/train/FedMLTrainer.cpp
        ${MOBILENN_HOME_DIR}/src/main_torch_train.cpp)

# link libraries and include directories
add_executable(${TARGET} ${DEMO_SRC})
target_include_directories(${TARGET} PRIVATE
        ${MOBILENN_HOME_DIR}/includes
        ${MOBILENN_HOME_DIR}/includes/torch
        ${MOBILENN_HOME_DIR}/includes/train)

if(${BUILD_ANDROID})
    target_include_directories(${TARGET} PRIVATE
            ${PYTORCH_HOME_DIR}/build_android/install/include
            ${PYTORCH_HOME_DIR}/build_android/install/include/torch/csrc/api/include
            ${PYTORCH_HOME_DIR}/aten/src
            ${PYTORCH_HOME_DIR}/include)
    target_link_libraries(${TARGET} PRIVATE
            ${PYTORCH_HOME_DIR}/build_android/install/lib/libc10.so
            ${PYTORCH_HOME_DIR}/build_android/install/lib/libtorch_cpu.so
            ${PYTORCH_HOME_DIR}/build_android/install/lib/libtorch_global_deps.so
            ${PYTORCH_HOME_DIR}/build_android/install/lib/libtorch.so
            log)
else()
    target_include_directories(${TARGET} PRIVATE
            ${PYTORCH_HOME_DIR}/build_mobile/install/include
            ${PYTORCH_HOME_DIR}/build_mobile/install/include/torch/csrc/api/include
            ${PYTORCH_HOME_DIR}/aten/src
            ${PYTORCH_HOME_DIR}/include)
    target_link_libraries(${TARGET} PRIVATE
            ${PYTORCH_HOME_DIR}/build_mobile/install/lib/libc10.dylib
            ${PYTORCH_HOME_DIR}/build_mobile/install/lib/libtorch_global_deps.dylib
            ${PYTORCH_HOME_DIR}/build_mobile/install/lib/libtorch.dylib
            ${PYTORCH_HOME_DIR}/build_mobile/install/lib/libtorch_cpu.dylib)
endif()
