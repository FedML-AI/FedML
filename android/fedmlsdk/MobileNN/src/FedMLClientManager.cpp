#include "FedMLClientManager.h"

FedMLClientManager::FedMLClientManager() {
    this->mFedMLTrainer = FedMLTrainer().getTrainer();  // input 1 to use MNN as backend
}

FedMLClientManager::~FedMLClientManager() {
    delete mFedMLTrainer;
}

void FedMLClientManager::init(const char *modelCachePath, const char *dataCachePath, const char *dataSet,
                                int trainSize, int testSize,
                                int batchSizeNum, double LearningRate, int epochNum,
                                progressCallback progress_callback,
                                accuracyCallback accuracy_callback,
                                lossCallback loss_callback) {
    this->mFedMLTrainer->init(modelCachePath, dataCachePath, dataSet, trainSize, testSize,
                              batchSizeNum, LearningRate, epochNum,
                              progress_callback, accuracy_callback, loss_callback);

//    this->m_modelCachePath = modelCachePath;
//    this->m_dataSet = dataSet;
}

std::string FedMLClientManager::train() {
    std::string result = this->mFedMLTrainer->train();
    return result;
}

std::string FedMLClientManager::getEpochAndLoss() {
    std::string result = this->mFedMLTrainer->getEpochAndLoss();
    return result;
}

bool FedMLClientManager::stopTraining() {
    bool result = this->mFedMLTrainer->stopTraining();
    return result;
}
