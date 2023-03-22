#ifndef FEDML_ANDROID_FEDMLTRAINER_H
#define FEDML_ANDROID_FEDMLTRAINER_H

#include "FedMLBaseTrainer.h"

#ifdef USE_MNN_BACKEND
#include "FedMLMNNTrainer.h"
#endif
#ifdef USE_TORCH_BACKEND
#include "FedMLTorchTrainer.h"
#endif


class FedMLTrainer {
    public:
        FedMLTrainer();
        FedMLBaseTrainer* getTrainer() {return m_trainer;}
        std::string getEpochAndLoss() {return m_trainer->getEpochAndLoss();}

    private:
        FedMLBaseTrainer* m_trainer;
};

#endif //FEDML_ANDROID_FEDMLTRAINER_H