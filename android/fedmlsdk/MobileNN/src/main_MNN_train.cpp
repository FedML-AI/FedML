#include <iostream>
#include <cstring>
#include <chrono>
#include <stdexcept>
#include "FedMLTrainer.h"


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
    std::cout << "You have entered " << argc
         << " arguments:" << "\n";

    for (int i = 0; i < argc; ++i)
        std::cout << argv[i] << "\n";

    /*
     * usage:
     *  ./build_x86_linux/main_mnn_train.out mnist ../../../../data/mnn_model/lenet_mnist.mnn ../../../../data/MNIST/raw
     */
    const char* datasetName = argv[1];
    const char* modelPath = argv[2];
    const char* dataPath = argv[3];


    int trainSize = 60000;
    int testSize = 10000;
    int batchSize = 8;
    double learningRate = 0.01;
    int epochNum = 10;

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
