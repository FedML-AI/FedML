cmake_minimum_required(VERSION 3.4.1)

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--as-needed")
set(LIB_DIR ${CMAKE_SOURCE_DIR}/../../../libs)
set(MOBILENN_DIR ${CMAKE_SOURCE_DIR}/../../../MobileNN)
set(MNN_DIR ${CMAKE_SOURCE_DIR}/../../../MobileNN/MNN)
set(TORCH_DIR ${CMAKE_SOURCE_DIR}/../../../MobileNN/pytorch)

if(${MOBILE_BACKEND} MATCHES MNN)
    message("########################## USING MNN BACKEND ##########################")
    add_definitions(-DUSE_MNN_BACKEND)
    add_definitions(-DMNN_USE_LOGCAT)
    # MNN add-on headers
    include_directories(${MOBILENN_DIR}/includes/MNN)
    # MNN add-on cpp
    file(GLOB_RECURSE MNN_SRCS ${MOBILENN_DIR}/src/MNN/*.cpp)

    # MNN source header
    include_directories(${MNN_DIR}/include)
    include_directories(${MNN_DIR}/tools/train/source/data)
    include_directories(${MNN_DIR}/tools/train/source/datasets)
    include_directories(${MNN_DIR}/tools/train/source/parameters)
    include_directories(${MNN_DIR}/tools/train/source/optimizer)
    include_directories(${MNN_DIR}/tools/train/source/nn)
    include_directories(${MNN_DIR}/tools/train/source/models)
    include_directories(${MNN_DIR}/tools/train/source/transformer)
    include_directories(${MNN_DIR}/tools/train/source/grad)

    # MNN source libraries
    add_library(MNN SHARED IMPORTED)
    set_target_properties(
            MNN
            PROPERTIES IMPORTED_LOCATION
            ${LIB_DIR}/${MOBILE_BACKEND}/${ANDROID_ABI}/libMNN.so
    )
    add_library(MNNTrain SHARED IMPORTED)
    set_target_properties(
            MNNTrain
            PROPERTIES IMPORTED_LOCATION
            ${LIB_DIR}/${MOBILE_BACKEND}/${ANDROID_ABI}/libMNNTrain.so
    )
    add_library(MNN_Express SHARED IMPORTED)
    list(APPEND MNN_LIBS MNN MNNTrain MNN_Express)
elseif(${MOBILE_BACKEND} MATCHES TORCH)
    message("########################## USING TORCH BACKEND ##########################")
#    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_THREAD_LIBS_INIT "-lpthread")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
    set(CMAKE_HAVE_THREADS_LIBRARY 1)
    set(CMAKE_USE_WIN32_THREADS_INIT 0)
    set(CMAKE_USE_PTHREADS_INIT 1)
    set(THREADS_PREFER_PTHREAD_FLAG ON)

    add_definitions(-DUSE_TORCH_BACKEND)

    # torch add-on headers
    include_directories(${MOBILENN_DIR}/includes/torch)
    # torch add-on cpp
    file(GLOB_RECURSE TORCH_SRCS ${MOBILENN_DIR}/src/torch/*.cpp)
    # torch source headers
    include_directories(${TORCH_DIR}/build_android/install/include)
    include_directories(${TORCH_DIR}/build_android/install/include/torch/csrc/api/include)
    include_directories(${TORCH_DIR}/aten/src)
    include_directories(${TORCH_DIR}/include)
    # torch source libraries
    add_library(c10 SHARED IMPORTED)
    set_target_properties(
            c10
            PROPERTIES IMPORTED_LOCATION
            ${LIB_DIR}/${MOBILE_BACKEND}/${ANDROID_ABI}/libc10.so
    )
    add_library(torch_cpu SHARED IMPORTED)
    set_target_properties(
            torch_cpu
            PROPERTIES IMPORTED_LOCATION
            ${LIB_DIR}/${MOBILE_BACKEND}/${ANDROID_ABI}/libtorch_cpu.so
    )
    add_library(torch_global_deps SHARED IMPORTED)
    set_target_properties(
            torch_global_deps
            PROPERTIES IMPORTED_LOCATION
            ${LIB_DIR}/${MOBILE_BACKEND}/${ANDROID_ABI}/libtorch_global_deps.so
    )
    add_library(torch SHARED IMPORTED)
    set_target_properties(
            torch
            PROPERTIES IMPORTED_LOCATION
            ${LIB_DIR}/${MOBILE_BACKEND}/${ANDROID_ABI}/libtorch.so
    )
    list(APPEND TORCH_LIBS c10 torch_cpu torch_global_deps torch)
endif()

# MobileNN headers
include_directories(${MOBILENN_DIR}/includes/train)
include_directories(${MOBILENN_DIR}/includes)
# MobileNN source cpp
file(GLOB_RECURSE MOBILENN_SRCS
        ${MOBILENN_DIR}/src/train/FedMLBaseTrainer.cpp
        ${MOBILENN_DIR}/src/train/FedMLTrainer.cpp
        ${MOBILENN_DIR}/src/FedMLClientManager.cpp)
if(${MOBILE_BACKEND} MATCHES MNN)
    file(GLOB_RECURSE MOBILENN_SRCS
        ${MOBILENN_DIR}/src/train/FedMLMNNTrainer.cpp)
elseif(${MOBILE_BACKEND} MATCHES TORCH)
    file(GLOB_RECURSE MOBILENN_SRCS
        ${MOBILENN_DIR}/src/train/FedMLTorchTrainer.cpp)
endif()

# SDK JNI headers
include_directories(${CMAKE_SOURCE_DIR}/includes)
# SDK JNI source cpp
file(GLOB_RECURSE MOBILENN_SRCS ${CMAKE_SOURCE_DIR}/*.cpp)

# MobileNN library
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${LIB_DIR}/${MOBILE_BACKEND}/${ANDROID_ABI}")
add_library(MobileNN SHARED ${MOBILENN_SRCS} ${MNN_SRCS} ${TROCH_SRCS})

# other denpendencies
find_library(log-lib log)
find_library(jnigraphics-lib jnigraphics)
target_link_libraries(MobileNN ${MNN_LIBS} ${TORCH_LIBS} ${log-lib} ${jnigraphics-lib})