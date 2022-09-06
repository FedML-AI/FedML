#ifndef FEDML_ANDROID_FEDMLTRAINERSA_H
#define FEDML_ANDROID_FEDMLTRAINERSA_H

#include <cstring>
#include <iostream>
#include "mnist.h"
#include "cifar10.h"
#include <MNN/expr/Executor.hpp>
#include "SGD.hpp"
#include <MNN/AutoTime.hpp>
#include "Loss.hpp"
#include "LearningRateScheduler.hpp"
#include "Transformer.hpp"
#include "NN.hpp"
#include "LightSecAggForMNN.h"

typedef std::function<void(float)> progressCallback;

typedef std::function<void(int, float)> accuracyCallback;

typedef std::function<void(int, float)> lossCallback;

class FedMLTrainerSA {

public:
    void init(const char *modelCachePath, const char *dataCachePath,
              const char *dataSet, int trainSize,
              int testSize, int batchSizeNum, double learningRate, int epochNum,
              progressCallback progress_callback,
              accuracyCallback accuracy_callback,
              lossCallback loss_callback);

    std::string train();

    std::string getEpochAndLoss();

    bool stopTraining();

private:
    std::string m_modelCachePath;
    std::string m_dataCachePath;
    std::string m_dataSet;
    int m_trainSize;
    int m_testSize;
    int m_batchSizeNum;
    double m_LearningRate;
    int m_epochNum;

    progressCallback m_progress_callback;
    accuracyCallback m_accuracy_callback;
    lossCallback m_loss_callback;
};


#endif //FEDML_ANDROID_FEDMLTRAINERSA_H
