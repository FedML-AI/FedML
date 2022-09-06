cmake_minimum_required(VERSION 3.0)

set (CMAKE_CXX_STANDARD 11)
set (TARTGET "main_fedml_client_mangaer.out")

# path to  MobileNN directory
set(MOBILENN_HOME_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../MobileNN")

# path to MNN directory
set(MNN_HOME_DIR "${CMAKE_CURRENT_LIST_DIR}/../../MNN")

# execute cmake in MNN root folder
add_subdirectory(${MNN_HOME_DIR} mnn_binary_dir)

# Source code path
file(GLOB_RECURSE DEMO_SRC
        ${MOBILENN_HOME_DIR}/src/MNN/*.cpp
        ${MOBILENN_HOME_DIR}/src/train/*.cpp
        ${MOBILENN_HOME_DIR}/src/FedMLClientManager.cpp
        ${MOBILENN_HOME_DIR}/src/main_FedMLClientManager.cpp)

# link libraries and include directories
add_executable(${TARTGET} ${DEMO_SRC})
target_link_libraries(${TARTGET} PRIVATE ${MNN_DEPS})
target_link_libraries(${TARTGET} PRIVATE MNNTrain)

target_include_directories(${TARTGET} PRIVATE ${MNN_HOME_DIR}/include)
target_include_directories(${TARTGET} PRIVATE ${MNN_HOME_DIR}/tools/train/source/grad)
target_include_directories(${TARTGET} PRIVATE ${MNN_HOME_DIR}/tools/train/source/optimizer)
target_include_directories(${TARTGET} PRIVATE ${MNN_HOME_DIR}/tools/train/source/transformer)
target_include_directories(${TARTGET} PRIVATE ${MNN_HOME_DIR}/tools/train/source/data)
target_include_directories(${TARTGET} PRIVATE ${MNN_HOME_DIR}/tools/train/source/nn)
target_include_directories(${TARTGET} PRIVATE ${MNN_HOME_DIR}/tools/train/source/models)
target_include_directories(${TARTGET} PRIVATE ${MNN_HOME_DIR}/tools/train/source/datasets)

target_include_directories(${TARTGET} PRIVATE
        ${MOBILENN_HOME_DIR}/includes
        ${MOBILENN_HOME_DIR}/includes/MNN
        ${MOBILENN_HOME_DIR}/includes/security
        ${MOBILENN_HOME_DIR}/includes/train)

