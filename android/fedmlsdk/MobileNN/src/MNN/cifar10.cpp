#include "cifar10.h"
#include <string.h>
#include <fstream>
#include <string>

namespace MNN {
namespace Train {

static int32_t kTrainSize;
static int32_t kTestSize;
static uint32_t kSizePerBatch;
const uint32_t kImageRows             = 32;
const uint32_t kImageColumns          = 32;
const uint32_t kBytesPerRow           = 3073;
const uint32_t kBytesPerChannelPerRow = 1024;

const std::vector<std::string> kTrainDataBatchFiles = {
    "data_batch_1.bin",
    "data_batch_2.bin",
    "data_batch_3.bin",
    "data_batch_4.bin",
    "data_batch_5.bin",
};

const std::vector<std::string> kTestDataBatchFiles = {
    "test_batch.bin"
};

std::pair<VARP, VARP> read_data(const std::string& root, bool train) {
    const auto& files = train ? kTrainDataBatchFiles : kTestDataBatchFiles;
    const auto num_samples = train ? kTrainSize : kTestSize;

    uint32_t kBytesPerBatchFile     = kBytesPerRow * kSizePerBatch;

    std::vector<char> data_buffer;
    data_buffer.reserve(files.size() * kBytesPerBatchFile);

    for (const auto& file : files) {
        auto path = root;
        if (path.back() != '/') {
            path.push_back('/');
        }
        path += file;
        std::ifstream data(path, std::ios::binary);
        if (!data.is_open()) {
            MNN_PRINT("Error opening data file at %s", path.c_str());
            MNN_ASSERT(false);
        }

        data_buffer.insert(data_buffer.end(), std::istreambuf_iterator<char>(data), {});
    }

    MNN_ASSERT(data_buffer.size() == files.size() * kBytesPerBatchFile);

    auto images = _Input({num_samples, 3, kImageRows, kImageColumns}, NCHW, halide_type_of<uint8_t>());
    auto labels = _Input({num_samples}, NCHW, halide_type_of<uint8_t>());

    for (uint32_t i = 0; i != num_samples; ++i) {
        // The first byte of each row is the target class index.
        uint32_t start_index = i * kBytesPerRow;
        labels->writeMap<uint8_t>()[i] = data_buffer[start_index];

        // The next bytes correspond to the rgb channel values in the following order:
        // red (32 *32 = 1024 bytes) | green (1024 bytes) | blue (1024 bytes)
        uint32_t image_start = start_index + 1;
        uint32_t image_end = image_start + 3 * kBytesPerChannelPerRow;
        std::copy(data_buffer.begin() + image_start, data_buffer.begin() + image_end,
            reinterpret_cast<char*>(images->writeMap<uint8_t>() + (i * 3 * kBytesPerChannelPerRow)));
    }

    return {images, labels};
}

Cifar10Dataset::Cifar10Dataset(const std::string root, Mode mode) {
    auto data = read_data(root, mode == Mode::TRAIN);
    mImages = data.first;
    mLabels = data.second;
    mImagePtr  = mImages->readMap<uint8_t>();
    mLabelsPtr = mLabels->readMap<uint8_t>();
}

Example Cifar10Dataset::get(size_t index) {
    auto data  = _Input({3, kImageRows, kImageColumns}, NCHW, halide_type_of<uint8_t>());
    auto label = _Input({}, NCHW, halide_type_of<uint8_t>());

    auto dataPtr = mImagePtr + index * 3 * kImageRows * kImageColumns;
    ::memcpy(data->writeMap<uint8_t>(), dataPtr, 3 * kImageRows * kImageColumns);

    auto labelPtr = mLabelsPtr + index;
    ::memcpy(label->writeMap<uint8_t>(), labelPtr, 1);

    auto returnIndex = _Const(index);
    // return the index for test
    return {{data, returnIndex}, {label}};
}

size_t Cifar10Dataset::size() {
    return mImages->getInfo()->dim[0];
}

const VARP Cifar10Dataset::images() {
    return mImages;
}

const VARP Cifar10Dataset::labels() {
    return mLabels;
}

DatasetPtr Cifar10Dataset::create(const std::string path, Mode mode, int32_t trainSize, int32_t testSize) {
    kTrainSize = trainSize;
    kTestSize = testSize;

    DatasetPtr res;
    res.mDataset.reset(new Cifar10Dataset(path, mode));
    return res;
}

}
}
