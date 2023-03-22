#include "FedMLClientManager.h"

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
     *  ./build_x86_linux/main_fedml_client_mangaer.out mnist ../../../../data/lenet_mnist.mnn ../../../../data/MNIST/raw
     */
    const char* datasetName = argv[1];
    const char* modelPath = argv[2];
    const char* dataPath = argv[3];

    int trainSize = 600;
    int testSize = 100;
    int batchSize = 8;
    double learningRate = 0.01;
    int epochNum = 1;

    MobileNNBackend backend = USE_TORCH;
    FedMLClientManager *mFedMLClientManager = new FedMLClientManager(backend);

    mFedMLClientManager->init(modelPath, dataPath, datasetName,
                              trainSize, testSize, batchSize, learningRate, epochNum,
                              std::bind(&onProgressCallback, std::placeholders::_1),
                              std::bind(&onLossCallback, std::placeholders::_1, std::placeholders::_2),
                              std::bind(&onAccuracyCallback, std::placeholders::_1, std::placeholders::_2));

    mFedMLClientManager->train();
    std::cout << mFedMLClientManager->getEpochAndLoss() << std::endl;
    return 0;
}