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
#include "string.h"

#define TAG "FedMLMNNTrainer"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Train;

class FedMLMNNTrainer: public FedMLBaseTrainer {
    public:
        std::string train() override;
};


#endif //FEDML_ANDROID_FEDMLMNNTRAINER_H
