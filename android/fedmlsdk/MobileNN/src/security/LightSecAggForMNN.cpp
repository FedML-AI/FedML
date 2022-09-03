#include "LightSecAggForMNN.h"

using namespace MNN;
using namespace MNN::Train;
using namespace MNN::Express;


void LightSecAggForMNN::init(int q_bits, int p, int client_num) {
    this->m_q_bits = q_bits;
    this->m_p = p;
    this->m_client_num = client_num;
}


void LightSecAggForMNN::printVar(VARP x) {
    auto size = x->getInfo()->size;
    auto ptr = x->readMap<float>();
    for (int i = 0; i < size; ++i) {
        MNN_PRINT("%f, ", ptr[i]);
    }
    MNN_PRINT("\n");
}


VARPS LightSecAggForMNN::mask_generate(const char *modelCachePath) {
    auto varMap = Variable::loadMap(modelCachePath);
    auto inputOutputs = Variable::getInputAndOutput(varMap);
    auto inputs = Variable::mapToSequence(inputOutputs.first);
    auto outputs = Variable::mapToSequence(inputOutputs.second);
    LightSecAggForMNN mec;
    // convert to trainable module and get weights
    std::shared_ptr <Module> model(NN::extract(inputs, outputs, true));
    auto param = model->parameters();

    // generate the mask for the model
    VARPS finite_weights = mec.transform_tensor_to_finite(param);
    VARPS model_mask = mec.generate_random_mask(finite_weights);

    return model_mask;
}


std::vector <std::vector<float>> LightSecAggForMNN::local_mask_encoding(VARPS model_mask) {
    std::vector<float> local_mask = mask_transform(model_mask);
    std::vector <std::vector<float>> encode_mask;
    encode_mask = mask_encoding(this->m_client_num, this->m_p, local_mask);
    return encode_mask;
}


void
LightSecAggForMNN::MNN_encode(const char *modelCachePath, const char *dataSet, VARPS model_mask) {

    auto varMap = Variable::loadMap(modelCachePath);
    auto inputOutputs = Variable::getInputAndOutput(varMap);
    auto inputs = Variable::mapToSequence(inputOutputs.first);
    auto outputs = Variable::mapToSequence(inputOutputs.second);

    // convert to trainable module and get weights
    std::shared_ptr <Module> model(NN::extract(inputs, outputs, true));
    auto param = model->parameters();

    // encode the model
    VARPS finite_weights = transform_tensor_to_finite(param);
    // mec.printVar(finite_weights[0]);
    model_masking(finite_weights, model_mask, this->m_p);
    model->loadParameters(finite_weights);

    //save the model
    VARP forwardInput;
    if (strcmp(dataSet, "mnist") == 0) {  // mnist dataset
        forwardInput = _Input({1, 1, 28, 28}, NC4HW4);
    } else if (strcmp(dataSet, "cifar10") == 0) { // cifar10 dataset
        forwardInput = _Input({1, 3, 32, 32}, NC4HW4);
    }
    model->setIsTraining(true);  // save the training state computation graph
    forwardInput->setName("data");
    auto inputPredict = model->forward(forwardInput);
    inputPredict->setName("prob");
    Variable::save({inputPredict}, modelCachePath);
    printf("masked model save done");
}


std::vector<float> LightSecAggForMNN::mask_agg(std::vector<int> surviving_list_from_server) {
    std::vector<float> sum_mask(this->m_local_received_mask_from_other_clients[0].size(), 0);
    for (int j = 0; j < sum_mask.size(); j++) {
        for (int i = 0; i < this->m_surviving_clients.size(); i++) {
            if(std::find(surviving_list_from_server.begin(), surviving_list_from_server.end(), this->m_surviving_clients[i]) != surviving_list_from_server.end() )
                sum_mask[j] = sum_mask[j] + this->m_local_received_mask_from_other_clients[i][j];
    }
    }
    return sum_mask;
}

void LightSecAggForMNN::save_mask_from_paired_clients(int client_index,
                                                      std::vector<float> received_encode_mask) {
    this->m_local_received_mask_from_other_clients.push_back(received_encode_mask);
    this->m_surviving_clients.push_back(client_index);
}

std::vector<int> LightSecAggForMNN::get_client_IDs_that_have_sent_mask() {
    return this->m_surviving_clients;
}

//private methods:

VARP LightSecAggForMNN::my_q(VARP const &X) {
    VARP result = _Input(X->getInfo()->dim, X->getInfo()->order, halide_type_of<float>());
    auto size = X->getInfo()->size;
    auto x = X->readMap<float>();
    auto y = result->writeMap<float>();
    for (int i = 0; i < size; ++i) {
        y[i] = round(x[i] * pow(2, this->m_q_bits));
        if (y[i] < 0) {
            y[i] += this->m_p;
        }
    }
    return result;
}

VARPS LightSecAggForMNN::transform_tensor_to_finite(VARPS const &model_params) {
    VARPS result;
    for (auto tmp: model_params) {
        tmp = my_q(tmp);
        result.push_back(tmp);
    }
    return result;
}

VARPS LightSecAggForMNN::generate_random_mask(VARPS const &model_params) {
    VARPS model_mask;
    for (auto tmp: model_params) {
        VARP layer_weights = _Input(tmp->getInfo()->dim, tmp->getInfo()->order, halide_type_of<float>());
        auto size = layer_weights->getInfo()->size;
        auto mask = layer_weights->writeMap<float>();
        for (int i = 0; i < size; ++i) {
            mask[i] = rand() % this->m_p;
        }
        model_mask.push_back(layer_weights);
    }
    return model_mask;
}


void LightSecAggForMNN::model_masking(VARPS &weights_finite, VARPS const &local_mask, int prime_number) {
    std::vector<int> g;
    int l = 1;
    g.push_back(l);
    VARP p = _Input(g, weights_finite[0]->getInfo()->order, halide_type_of<float>());
    auto prime = p->writeMap<float>();
    prime[0] = prime_number;
    for (int i = 0; i < weights_finite.size(); i++) {
        weights_finite[i] = weights_finite[i] + local_mask[i];
        weights_finite[i] = _FloorMod(weights_finite[i], p);
    }

}

std::vector<float> LightSecAggForMNN::mask_transform(VARPS model_mask) {
    std::vector<float> local_mask;
    for (auto tmp: model_mask) {
        auto size = tmp->getInfo()->size;
        auto x = tmp->readMap<float>();
        for (int i = 0; i < size; ++i) {
            local_mask.push_back(x[i]);
        }
    }
    return local_mask;
}

std::vector <std::vector<float>>
LightSecAggForMNN::mask_encoding(int num_clients, int prime_number, std::vector<float> const &local_mask) {
    int d = local_mask.size();
    int N = num_clients;
    int T = N / 2;
    int U = T + 1;
    int p = prime_number;

    std::vector<int> beta_s(N);
    std::iota(std::begin(beta_s), std::end(beta_s), 1);
    std::vector<int> alpha_s(U);
    std::iota(std::begin(alpha_s), std::end(alpha_s), N + 1);

    auto local_mask_rand = local_mask;
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<float> dist(1, p);
    for (int i = 0; i < (T * d / (U - T)); i++)
        local_mask_rand.push_back(dist(rng));

    int y = d / (U - T);
    std::vector <std::vector<float>> LCC_in(U, std::vector<float>(y));

    for (int i = 0; i < local_mask_rand.size(); i++)
        LCC_in[i / y][i % y] = local_mask_rand[i];
    LightSecAgg lsa;
    std::vector <std::vector<float>> encoded_mask_set = lsa.LCC_encoding_with_points(LCC_in, alpha_s, beta_s, p);
    return encoded_mask_set;
}

std::vector<float> LightSecAggForMNN::z_tilde_sum(std::vector <std::vector<float>> const &z_tilde_buffer,
                                                  std::vector<int> const &sur_list) {
    std::vector<float> w(z_tilde_buffer[0].size(), 0.0);
    for (int i = 0; i < w.size(); ++i) {
        for (int j: sur_list)
            w[i] = w[i] + z_tilde_buffer[i][j];
    }
    return w;
}

