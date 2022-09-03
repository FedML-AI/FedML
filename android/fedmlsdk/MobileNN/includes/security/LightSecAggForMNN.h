#ifndef FEDML_ANDROID_LIGHTSECAGGFORMNN_H
#define FEDML_ANDROID_LIGHTSECAGGFORMNN_H


#include <iostream>
#include <string.h>
#include <chrono>
#include <stdexcept>
#include "mnist.h"
#include "cifar10.h"
#include <MNN/expr/Executor.hpp>
#include "SGD.hpp"
#include <MNN/AutoTime.hpp>
#include "Loss.hpp"
#include "LearningRateScheduler.hpp"
#include "Transformer.hpp"
#include "NN.hpp"
#include "LightSecAgg.h"
#include <climits>
#include <algorithm> 

class LightSecAggForMNN {

public:
    void init(int q_bits, int p, int client_num);

    static void printVar(VARP x);

    VARPS mask_generate(const char *modelCachePath);

    std::vector <std::vector<float>> local_mask_encoding(VARPS model_mask);

    void MNN_encode(const char *modelCachePath, const char *dataSet, VARPS model_mask);

    std::vector<float> mask_agg(std::vector<int> surviving_list_from_server);

    void save_mask_from_paired_clients(int client_index,
                                       std::vector<float> local_encode_mask);

    std::vector<int> get_client_IDs_that_have_sent_mask();

private:

    VARP my_q(VARP const &X);

    VARPS transform_tensor_to_finite(VARPS const &model_params);

    VARPS generate_random_mask(VARPS const &model_params);

    void model_masking(VARPS &weights_finite, VARPS const &local_mask, int prime_number);

    std::vector<float> mask_transform(VARPS model_mask);

    std::vector <std::vector<float>>
    mask_encoding(int num_clients, int prime_number, std::vector<float> const &local_mask);

    std::vector<float>
    z_tilde_sum(std::vector <std::vector<float>> const &z_tilde_buffer, std::vector<int> const &sur_list);

private:
    int m_q_bits;
    int m_p;
    int m_client_num;

    std::vector <std::vector<float>> m_local_received_mask_from_other_clients;
    std::vector<int> m_surviving_clients;
};

#endif //FEDML_ANDROID_LIGHTSECAGGFORMNN_H
