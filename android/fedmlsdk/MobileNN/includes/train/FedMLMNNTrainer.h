#ifndef FEDML_ANDROID_FEDMLMNNTRAINER_H
#define FEDML_ANDROID_FEDMLMNNTRAINER_H

#include "FedMLBaseTrainer.h"
#include "mnist.h"
#include "cifar10.h"
#include <MNN/expr/Executor.hpp>
#include "DataLoader.hpp"
#include "SGD.hpp"
#include "Loss.hpp"
#include "LearningRateScheduler.hpp"
#include "Transformer.hpp"
#include "NN.hpp"

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Train;

class FedMLMNNTrainer: public FedMLBaseTrainer {
    public:
        std::string train() override;
};


#endif //FEDML_ANDROID_FEDMLMNNTRAINER_H
