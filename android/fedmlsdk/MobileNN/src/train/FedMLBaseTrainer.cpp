#include "FedMLBaseTrainer.h"

void FedMLBaseTrainer::init(const char *modelCachePath, const char *dataCachePath,
                        const char *dataSet, int trainSize, int testSize,
                        int batchSizeNum, double learningRate, int epochNum,
                        progressCallback progress_callback,
                        accuracyCallback accuracy_callback,
                        lossCallback loss_callback) {

    m_modelCachePath = modelCachePath;
    m_dataCachePath = dataCachePath;
    m_dataSet = dataSet;

    m_trainSize = trainSize;
    m_testSize = testSize;
    m_batchSizeNum = batchSizeNum;
    m_LearningRate = learningRate;
    m_epochNum = epochNum;

    m_progress_callback = progress_callback;
    m_accuracy_callback = accuracy_callback;
    m_loss_callback = loss_callback;
}

std::string FedMLBaseTrainer::getEpochAndLoss() {
    std::string result = std::to_string(curEpoch) + "," + std::to_string(curLoss);
    return result;
}

bool FedMLBaseTrainer::stopTraining() {
    bRunStopFlag = true;
    printf("stopTraining By User.");
    return true;
}