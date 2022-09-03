#ifndef FEDML_ANDROID_FEDMLTORCHTRAINER_H
#define FEDML_ANDROID_FEDMLTORCHTRAINER_H

#include "FedMLBaseTrainer.h"
#include <iostream>
#include "cifar10.h"
#include "mnist.h"

#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/mobile/train/export_data.h>
#include <torch/csrc/jit/mobile/train/optim/sgd.h>
#include <torch/csrc/jit/mobile/train/sequential.h>


class FedMLTorchTrainer : public FedMLBaseTrainer {
    public:
        std::string train() override;
};


#endif //FEDML_ANDROID_FEDMLTORCHTRAINER_H
