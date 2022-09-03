#include "FedMLClientManagerSA.h"

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Train;

FedMLClientManagerSA::FedMLClientManagerSA() {
    this->mFedMLTrainer = new FedMLTrainerSA();
    this->mLightSecAggForMNN = new LightSecAggForMNN();
}

FedMLClientManagerSA::~FedMLClientManagerSA() {
    delete mFedMLTrainer;
    delete mLightSecAggForMNN;
}

void FedMLClientManagerSA::init(const char *modelCachePath, const char *dataCachePath, const char *dataSet,
                                int trainSize, int testSize,
                                int batchSizeNum, double LearningRate, int epochNum,
                                int q_bits, int p, int client_num,
                                progressCallback progress_callback,
                                accuracyCallback accuracy_callback,
                                lossCallback loss_callback) {
    this->mFedMLTrainer->init(modelCachePath, dataCachePath, dataSet, trainSize, testSize,
                        batchSizeNum, LearningRate, epochNum,
                        progress_callback, accuracy_callback, loss_callback);

    this->mLightSecAggForMNN->init(q_bits, p, client_num);

    this->m_modelCachePath = modelCachePath;
    this->m_dataSet = dataSet;
}

std::vector <std::vector<float>> FedMLClientManagerSA::get_local_encoded_mask() {
    this->m_local_mask = this->mLightSecAggForMNN->mask_generate(this->m_modelCachePath.data());
    std::vector <std::vector<float>> local_encode_mask = this->mLightSecAggForMNN->local_mask_encoding(this->m_local_mask);
    return local_encode_mask;
}


void FedMLClientManagerSA::save_mask_from_paired_clients(int client_index,
                                                         std::vector<float> local_encode_mask) {

    this->mLightSecAggForMNN->save_mask_from_paired_clients(client_index,
                                                      local_encode_mask);

}

std::vector<int> FedMLClientManagerSA::get_client_IDs_that_have_sent_mask() {
    return this->mLightSecAggForMNN->get_client_IDs_that_have_sent_mask();
}

std::string FedMLClientManagerSA::train() {
    std::string result = this->mFedMLTrainer->train();
    return result;
}

void FedMLClientManagerSA::generate_masked_model() {
    this->mLightSecAggForMNN->MNN_encode(this->m_modelCachePath.data(), this->m_dataSet.data(), this->m_local_mask);
}

std::vector<float> FedMLClientManagerSA::get_aggregated_encoded_mask(std::vector<int> surviving_list_from_server) {
    std::vector<float> sum_mask = this->mLightSecAggForMNN->mask_agg(surviving_list_from_server);
    return sum_mask;
}


std::string FedMLClientManagerSA::getEpochAndLoss() {
    std::string result = this->mFedMLTrainer->getEpochAndLoss();
    return result;
}

bool FedMLClientManagerSA::stopTraining() {
    bool result = this->mFedMLTrainer->stopTraining();
    return result;
}

void FedMLClientManagerSA::printMNNVar(VARP x) {
    this->mLightSecAggForMNN->printVar(x);
}