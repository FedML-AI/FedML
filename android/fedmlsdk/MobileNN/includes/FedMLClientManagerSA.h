#ifndef FEDML_ANDROID_FEDMLCLIENTMANAGERSA_H
#define FEDML_ANDROID_FEDMLCLIENTMANAGERSA_H

#include "train/FedMLTrainerSA.h"
#include "LightSecAggForMNN.h"

class FedMLClientManagerSA {

public:
    FedMLClientManagerSA();

    ~FedMLClientManagerSA();

    void init(const char *modelCachePath, const char *dataCachePath, const char *dataSet,
              int trainSize, int testSize,
              int batchSizeNum, double LearningRate, int epochNum,
              int q_bits, int p, int client_num,
              progressCallback progress_callback,
              accuracyCallback accuracy_callback,
              lossCallback loss_callback);

    /**
     * generate local mask and encode mask to share with other users
     */
    std::vector <std::vector<float>> get_local_encoded_mask();

    /**
     * receive other mask from surviving users
     */
    void save_mask_from_paired_clients(int client_index,
                                       std::vector<float> local_encode_mask);


    /**
     * receive client index from surviving users
     */
    std::vector<int> get_client_IDs_that_have_sent_mask();


    std::string train();

    /**
     * get masked model after the local training is done
     * the model file is saved at the original path "modelCachePath"
     */
    void generate_masked_model();

    /**
     * the server will ask those clients that are online to send aggregated encoded masks
     */
    std::vector<float> get_aggregated_encoded_mask(std::vector<int> surviving_list_from_server);

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

    /**
     * print MNN variables
     */
    void printMNNVar(VARP x);

private:
    FedMLTrainerSA *mFedMLTrainer;
    LightSecAggForMNN *mLightSecAggForMNN;

    VARPS m_local_mask;
    std::string m_modelCachePath;
    std::string m_dataSet;
};


#endif //FEDML_ANDROID_FEDMLCLIENTMANAGERSA_H
