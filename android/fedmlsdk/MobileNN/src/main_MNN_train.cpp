#include <iostream>
#include <cstring>
#include <chrono>
#include <stdexcept>
#include "FedMLTrainer.h"


static void onProgressCallback(float progress) {
    // printf("callback. progress = %f\n", progress);
    // printf("")
    return ;
}

static void onLossCallback(int epoch, float loss) {
    // printf("callback. epoch = %d, loss = %f\n", epoch, loss);
    return ;
}

static void onAccuracyCallback(int epoch, float accuracy) {
    printf("callback. epoch = %d, accuracy = %f\n", epoch, accuracy);
}


int main(int argc, char *argv[]) {
    std::cout << "You have entered " << argc
         << " arguments:" << "\n";

    for (int i = 0; i < argc; ++i)
        std::cout << argv[i] << "\n";

    /*
     * usage:
     *  ./build_x86_linux/main_mnn_train.out mnist ../../../../data/mnn_model/lenet_mnist.mnn ../../../../data/MNIST/raw/client1
     *  ./main_mnn_train.out tabular ../../../../data/mnn_model/lenet_mnist.mnn ../../../../data/tabular
     */
    const char* datasetName = argv[1];
    const char* modelPath = argv[2];
    const char* dataPath = argv[3];


    int trainSize = 60000;
    int testSize = 10000;
    int batchSize = 8;
    double learningRate = 0.01;
    int epochNum = 10;

    #ifdef IS_DEBUG
    trainSize = 16;
    testSize = 16;
    batchSize = 1;
    learningRate = 0.01;
    epochNum = 1;
    #endif
    printf("trainSize = %d, testSize = %d, batchSize = %d, learningRate = %f, epochNum = %d\n",
            trainSize, testSize, batchSize, learningRate, epochNum);

    FedMLBaseTrainer *pFedMLTrainer = FedMLTrainer().getTrainer();
    pFedMLTrainer->init(modelPath, dataPath,
                    datasetName, trainSize, testSize,
                    batchSize, learningRate, epochNum,
                    std::bind(&onProgressCallback, std::placeholders::_1),
                    std::bind(&onAccuracyCallback, std::placeholders::_1, std::placeholders::_2),
                    std::bind(&onLossCallback, std::placeholders::_1, std::placeholders::_2));
    pFedMLTrainer->train();

    return 0;

}
