//
//  MnistDataset.hpp
//  MNN
//
//  Created by MNN on 2019/11/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MnistDataset_hpp
#define MnistDataset_hpp

#include <string>
#include "Dataset.hpp"
#include "Example.hpp"

namespace MNN {
namespace Train {
class MNN_PUBLIC MnistDataset : public Dataset {
public:
    enum Mode { TRAIN, TEST };

    Example get(size_t index) override;

    size_t size() override;

    const VARP images();

    const VARP labels();

    static DatasetPtr create(const std::string path, Mode mode = Mode::TRAIN, int32_t trainSize = 60000, int32_t testSize = 10000);
private:
    explicit MnistDataset(const std::string path, Mode mode = Mode::TRAIN);
    VARP mImages, mLabels;
    const uint8_t* mImagePtr  = nullptr;
    const uint8_t* mLabelsPtr = nullptr;
};
}
}


#endif // MnistDataset_hpp
