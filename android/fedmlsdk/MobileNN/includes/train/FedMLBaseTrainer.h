#ifndef FEDML_ANDROID_FEDMLBASETRAINER_H
#define FEDML_ANDROID_FEDMLBASETRAINER_H

#include <cstring>
#include <iostream>

typedef std::function<void(float)> progressCallback;

typedef std::function<void(int, float)> accuracyCallback;

typedef std::function<void(int, float)> lossCallback;

class FedMLBaseTrainer {

public:
    void init(const char *modelCachePath, const char *dataCachePath,
              const char *dataSet, int trainSize, int testSize,
              int batchSizeNum, double learningRate, int epochNum,
              progressCallback progress_callback,
              accuracyCallback accuracy_callback,
              lossCallback loss_callback);

    virtual std::string train() {return nullptr;};

    std::string getEpochAndLoss();

    bool stopTraining();

protected:
    std::string m_modelCachePath;
    std::string m_dataCachePath;
    std::string m_dataSet;
    int m_trainSize;
    int m_testSize;
    int m_batchSizeNum;
    double m_LearningRate;
    int m_epochNum;

    int curEpoch = 0;
    float curLoss = 0.0;
    bool bRunStopFlag = false;

    progressCallback m_progress_callback;
    accuracyCallback m_accuracy_callback;
    lossCallback m_loss_callback;
};


#endif //FEDML_ANDROID_FEDMLBASETRAINER_H
