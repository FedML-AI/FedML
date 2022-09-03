#ifndef Cifar10Dateset_hpp
#define Cifar10Dateset_hpp

#include <string>
#include "Dataset.hpp"
#include "Example.hpp"

namespace MNN {
namespace Train {
class MNN_PUBLIC Cifar10Dataset : public Dataset {
public:
    enum Mode { TRAIN, TEST };

    Example get(size_t index) override;

    size_t size() override;

    const VARP images();

    const VARP labels();

    static DatasetPtr create(const std::string path, Mode mode = Mode::TRAIN, int32_t trainSize = 50000, int32_t testSize = 10000);
private:
    explicit Cifar10Dataset(const std::string path, Mode mode = Mode::TRAIN);
    VARP mImages, mLabels;
    const uint8_t* mImagePtr  = nullptr;
    const uint8_t* mLabelsPtr = nullptr;
};
}
}


#endif // Cifar10Dateset_hpp