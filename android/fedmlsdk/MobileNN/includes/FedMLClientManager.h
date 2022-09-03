#ifndef FEDML_ANDROID_FEDMLCLIENTMANAGER_H
#define FEDML_ANDROID_FEDMLCLIENTMANAGER_H

#include "FedMLTrainer.h"

class FedMLClientManager {

public:
    FedMLClientManager();

    ~FedMLClientManager();

    void init(const char *modelCachePath, const char *dataCachePath, const char *dataSet,
              int trainSize, int testSize, int batchSizeNum, double LearningRate, int epochNum,
              progressCallback progress_callback,
              accuracyCallback accuracy_callback,
              lossCallback loss_callback);

    std::string train();

    /**
     * the local epoch index in each global epoch training, and the training loss in this local epoch
     *
     * @return current epoch and the loss value in this epoch (format: "epoch,loss")
     */
    std::string getEpochAndLoss();

    /**
     * Stop the current training
     *
     * @return success
     */
    bool stopTraining();

private:
    FedMLBaseTrainer *mFedMLTrainer;

//    std::string m_modelCachePath;
//    std::string m_dataSet;
};


#endif //FEDML_ANDROID_FEDMLCLIENTMANAGER_H
