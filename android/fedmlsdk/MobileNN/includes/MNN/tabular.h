#ifndef TabularDataset_hpp
#define TabularDataset_hpp

#include <string>
#include "Dataset.hpp"
#include "Example.hpp"

namespace MNN {
namespace Train {
class MNN_PUBLIC TabularDataset: public Dataset {
public:
    enum Mode { TRAIN, TEST };

    Example get(size_t index) override;

    size_t size() override;

    const VARP features();

    const VARP labels();

    static DatasetPtr create(const std::string path, Mode mode = Mode::TRAIN, int32_t trainSize = 4, int32_t testSize = 2);
private:
    explicit TabularDataset(const std::string path, Mode mode = Mode::TRAIN);
    VARP mFeatures, mLabels;
    const float* mFeaturesPtr  = nullptr;
    const float* mLabelsPtr = nullptr;
};
}
}


#endif // TabularDataset_hpp
