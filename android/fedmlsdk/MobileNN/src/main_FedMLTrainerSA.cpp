#include "FedMLClientManagerSA.h"
#include <math.h>



static void onProgressCallback(float progress) {
    printf("callback. progress = %f\n", progress);
}

static void onLossCallback(int epoch, float loss) {
    printf("callback. epoch = %d, loss = %f\n", epoch, loss);
}

static void onAccuracyCallback(int epoch, float accuracy) {
    printf("callback. epoch = %d, accuracy = %f\n", epoch, accuracy);
}


int main(int argc, char *argv[]) {
    std::cout << "You have entered 1" << argc
              << " arguments:" << "\n";

    for (int i = 0; i < argc; ++i)
        std::cout << argv[i] << "\n";

    /*
     * usage:
     *  ./build_x86_linux/fedml_trainer.out mnist ../../../../data/lenet_mnist.mnn ../../../../data/mnist
     */
    const char *dataSetType = argv[1];
    const char *modelPath = argv[2];
    const char *dataPath = argv[3];

    int epochNum = 1;
    float learningRate = 0.01;
    int batchSizeNum = 8;
    int trainSize = 20000;
    int testSize = 1000;

    //test parameter for the encoding part
    int client_num = 10;
    int q_bits = 15;
    int p = pow(2, 15) - 19;
    printf("debug22");
    printf("main::CreateModelFromFile(%s, %s, %s, %d, %f, %d, %d, %d)\n", modelPath, dataPath, dataSetType,
           batchSizeNum,
           learningRate, epochNum, trainSize, testSize);
    printf("debug11");
    FedMLClientManager *mFedMLClientManager = new FedMLClientManager();

    //init all required parameters
    printf("debug1");

    mFedMLClientManager->init(modelPath, dataPath, dataSetType,
                              trainSize, testSize, batchSizeNum,
                              learningRate, epochNum,
                              q_bits, p, client_num,
                              std::bind(&onProgressCallback, std::placeholders::_1),
                              std::bind(&onLossCallback, std::placeholders::_1, std::placeholders::_2),
                              std::bind(&onAccuracyCallback, std::placeholders::_1, std::placeholders::_2));

    /**
     * 1. generate mask and encode local mask for others
     */
    printf("debug2");
    std::vector <std::vector<float>> encoded_mask = mFedMLClientManager->get_local_encoded_mask();
    std::cout << encoded_mask[0].size() << std::endl;

    /**
     * 2. share and receive local mask from others via server (including share to self)
     * call this function repeatedly during listening phase, once we receive a pair, store it via this function
     */
    printf("debug3");
    int client_index = 1;
    std::vector<float> local_encode_mask = encoded_mask[0];
    mFedMLClientManager->save_mask_from_paired_clients(client_index, local_encode_mask);

    int client_index_another = 3;
    std::vector<float> local_encode_mask_another = encoded_mask[9];
    mFedMLClientManager->save_mask_from_paired_clients(client_index_another, local_encode_mask_another);

    /**
     * 3. report receive online users to server
     */
    printf("debug4");
    std::vector<int> online_user = mFedMLClientManager->get_client_IDs_that_have_sent_mask();

    /**
     * 4. do training
     */
    printf("debug5");
    mFedMLClientManager->train();

    /**
     * 5. save masked model
     */
    printf("debug6");
    mFedMLClientManager->generate_masked_model();

    /**
     * 6. receive online user list from server
     * aggregate received mask
     * surviving list represents the users that is online confirmed by server
     */
    printf("debug7");
    std::vector<int> surviving_list_from_server;
    surviving_list_from_server.push_back(1);
    surviving_list_from_server.push_back(3);
    std::vector<float> upload_agg_mask = mFedMLClientManager->get_aggregated_encoded_mask(surviving_list_from_server);
    std::cout << upload_agg_mask[0];

    /**
     * test loading function for the encoded mnn
     * 
     */
    // load computational graph
    auto varMap = Variable::loadMap(modelPath);
    auto inputOutputs = Variable::getInputAndOutput(varMap);
    auto inputs = Variable::mapToSequence(inputOutputs.first);
    auto outputs = Variable::mapToSequence(inputOutputs.second);

    // convert to trainable module
    std::shared_ptr <Module> model(NN::extract(inputs, outputs, true));

    return 0;
}