cmake_minimum_required(VERSION 3.0)

set (CMAKE_CXX_STANDARD 11)
set (TARGET "main_mnn_train.out")

# path to  MobileNN directory
set(MOBILENN_HOME_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../MobileNN")

# enable MNN backend
add_definitions(-DUSE_MNN_BACKEND)

# path to MNN directory
set(MNN_HOME_DIR "${MOBILENN_HOME_DIR}/MNN")

# execute cmake in MNN root folder
add_subdirectory(${MNN_HOME_DIR} mnn_binary_dir)

# Source code path
file(GLOB_RECURSE DEMO_SRC
        ${MOBILENN_HOME_DIR}/src/MNN/*.cpp
        ${MOBILENN_HOME_DIR}/src/train/FedMLBaseTrainer.cpp
        ${MOBILENN_HOME_DIR}/src/train/FedMLMNNTrainer.cpp
        ${MOBILENN_HOME_DIR}/src/train/FedMLTrainer.cpp
        ${MOBILENN_HOME_DIR}/src/main_MNN_train.cpp)

# link libraries and include directories
add_executable(${TARGET} ${DEMO_SRC})
target_link_libraries(${TARGET} PRIVATE ${MNN_DEPS})
target_link_libraries(${TARGET} PRIVATE MNNTrain)

target_include_directories(${TARGET} PRIVATE ${MNN_HOME_DIR}/include)
target_include_directories(${TARGET} PRIVATE ${MNN_HOME_DIR}/tools/train/source/grad)
target_include_directories(${TARGET} PRIVATE ${MNN_HOME_DIR}/tools/train/source/optimizer)
target_include_directories(${TARGET} PRIVATE ${MNN_HOME_DIR}/tools/train/source/transformer)
target_include_directories(${TARGET} PRIVATE ${MNN_HOME_DIR}/tools/train/source/data)
target_include_directories(${TARGET} PRIVATE ${MNN_HOME_DIR}/tools/train/source/nn)
target_include_directories(${TARGET} PRIVATE ${MNN_HOME_DIR}/tools/train/source/models)
target_include_directories(${TARGET} PRIVATE ${MNN_HOME_DIR}/tools/train/source/datasets)

target_include_directories(${TARGET} PRIVATE
        ${MOBILENN_HOME_DIR}/includes
        ${MOBILENN_HOME_DIR}/includes/MNN
        ${MOBILENN_HOME_DIR}/includes/train)

