#include "tabular.h"
#include <string.h>
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
namespace MNN {
namespace Train {

// referenced from pytorch C++ frontend mnist.cpp
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/src/data/datasets/mnist.cpp
static int32_t kTrainSize;  // init in create()
static int32_t kTestSize;   // init in create()
static uint32_t kSizePerBatch;
const int32_t FeaturesNum       = 28*28;
const uint32_t kBytesPerRow           = 3073;
const uint32_t kBytesPerChannelPerRow = 1024;
const std::vector<std::string> kTrainCsvFilenames  = {"mnist_train.csv"};
const std::vector<std::string> kTestCsvFilenames = {"mnist_test.csv"};

std::pair<VARP, VARP> read_csv (const std::string& root, bool train) {
    std::cout << "starting read csv ..." << std::endl;
    std::cout << "1. determin file names, number of samples in that file" << std::endl;
    const auto& files = train ? kTrainCsvFilenames : kTestCsvFilenames;
    const auto num_samples = train ? kTrainSize : kTestSize;

    std::cout << "2. prepare the buffer to read the data" << std::endl;
    std::vector<std::vector<float>> data_buffer;
    // The following code is for optimized reading
    // uint32_t kBytesPerBatchFile = kBytesPerRow * kSizePerBatch;
    // data_buffer.reserve(files.size() * kBytesPerBatchFile);

    std::cout << "3. read the data to the buffer" << std::endl;
    for (const auto & file : files) {
        auto path = root;
        if (path.back() != '/') {
            path.push_back('/');
        }
        path += file;   // root folder + file name

        std::ifstream data(path, std::ios::binary);
        if (!data.is_open()) {
            MNN_PRINT("Error opening images file at %s", path.c_str());
            MNN_ASSERT(false);
        }
        // read each row to data_buffer from that file stream
        std::string line;
        while (std::getline(data, line))
        {
            std::vector<float> row;
            std::string cell;
            std::stringstream lineStream(line);
            
            // Split line into cells
            while (std::getline(lineStream, cell, ',')) {
                try {
                    float cellValue = std::stof(cell);
                    row.push_back(cellValue);
                    // debug
                    // std::cout << "cell: " << cell << std::endl;
                    // std::cout << "cellValue: " << cellValue << std::endl;
                }
                catch (const std::invalid_argument& e) {
                    std::cerr << "Could not convert string to float: " << cell << std::endl;
                    exit(-1);
                }
            }
            data_buffer.push_back(row);    // e.g. "1,2,3,4,5,6"
        }
        data.close();
    }

    // Debug
    // for (auto& row : data_buffer) {
    //     for (auto& cell : row) {
    //         std::cout << cell << ' ';
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << "num_samples: " << num_samples << std::endl;
    // std::cout << "FeaturesNum: " << FeaturesNum << std::endl;

    std::cout << "4. Init the result mnn tensor using dimensions of the data" << std::endl;
    const auto count = train ? kTrainSize : kTestSize;
    std::vector<int> dims = {count, 1, 28, 28};
    auto features = _Input(dims, NCHW, halide_type_of<float>());
    auto labels = _Input({count}, NCHW, halide_type_of<float>());

    std::cout << "5. read the data, the first FeaturesNum are features, the last one is the label" << std::endl;
    // read the data from data_buffer, split by ",", then for the first FeaturesNum are features, the last one is the label, then write to the result mnn tensor
    for (int i = 0; i < num_samples; i++) {
        auto dataPtr = features->writeMap<float>() + i * FeaturesNum;
        auto labelPtr = labels->writeMap<float>() + i;
        // for (int j = 0; j < FeaturesNum; ++j) {
        //     // using std::copy to copy data from data_buffer[i][j] to dataPtr[j]
            
        //     printf("dataPtr[%d] = %f\n", j, dataPtr[j]);
        // }
        std::copy(data_buffer[i].begin(), data_buffer[i].begin() + FeaturesNum, dataPtr);
        labelPtr[0] = data_buffer[i][FeaturesNum];
    }

    //debug
    // const float *outputPtr = features->readMap<float>();
    // auto outputSize = features->getInfo()->size;
    // std::cout << "outputSize: " << outputSize << std::endl;
    // for (int i=0; i<outputSize; ++i) {
    //     printf("%f, ", outputPtr[i]);
    // }
    // exit(0);

    std::cout << "6. read csv done" << std::endl;
    return {features, labels};
}

TabularDataset::TabularDataset(const std::string root, Mode mode) {
    auto data = read_csv(root, mode == Mode::TRAIN);
    mFeatures = data.first;
    mLabels = data.second;

    mFeaturesPtr  = mFeatures->readMap<float>();
    mLabelsPtr = mLabels->readMap<float>();
}

Example TabularDataset::get(size_t index) {
    auto data  = _Input({1, 28, 28}, NCHW, halide_type_of<float>());
    auto label = _Input({}, NCHW, halide_type_of<float>());

    auto dataPtr = mFeaturesPtr + index * FeaturesNum;

    // Note: Here we do not prefer to use ::memcpy,
    // since we want to copy float type
    std::copy(dataPtr, dataPtr + FeaturesNum, data->writeMap<float>());

    auto labelPtr = mLabelsPtr + index;

    // Note: Here we do not prefer to use ::memcpy,
    // since we want to copy float type
    std::copy(labelPtr, labelPtr + 1, label->writeMap<float>());

    auto returnIndex = _Const(index);
    // return the index for test
    return {{data, returnIndex}, {label}};
}

size_t TabularDataset::size() {
    return mFeatures->getInfo()->dim[0];
}

const VARP TabularDataset::features() {
    return mFeatures;
}

const VARP TabularDataset::labels() {
    return mLabels;
}

DatasetPtr TabularDataset::create(const std::string path, Mode mode, int32_t trainSize, int32_t testSize) {
    kTrainSize = trainSize;
    kTestSize = testSize;

    DatasetPtr res;
    res.mDataset.reset(new TabularDataset(path, mode));
    return res;
}
}
}
