#include "FedMLTrainerSA.h"

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Train;

// Global variable to get the current epoch value and current loss in this epoch
static int curEpoch = 0;
static float curLoss = 0.0;
static bool bRunStopFlag = false;


void FedMLTrainerSA::init(const char *modelCachePath, const char *dataCachePath,
                        const char *dataSet, int trainSize,
                        int testSize, int batchSizeNum, double learningRate, int epochNum,
                        progressCallback progress_callback,
                        accuracyCallback accuracy_callback,
                        lossCallback loss_callback) {

    m_modelCachePath = modelCachePath;
    m_dataCachePath = dataCachePath;
    m_dataSet = dataSet;

    m_trainSize = trainSize;
    m_testSize = testSize;
    m_batchSizeNum = batchSizeNum;
    m_LearningRate = learningRate;
    m_epochNum = epochNum;

    m_progress_callback = progress_callback;
    m_accuracy_callback = accuracy_callback;
    m_loss_callback = loss_callback;
}

std::string FedMLTrainerSA::getEpochAndLoss() {
    std::string result = std::to_string(curEpoch) + "," + std::to_string(curLoss);
    return result;
}

bool FedMLTrainerSA::stopTraining() {
    bRunStopFlag = true;
    printf("stopTraining By User.");
    return true;
}


std::string FedMLTrainerSA::train() {

    bRunStopFlag = false;

    const char *modelPath = m_modelCachePath.data();
    const char *dataPath = m_dataCachePath.data();
    const char *dataSetType = m_dataSet.data();

    int trainSize = m_trainSize;
    int testSize = m_testSize;
    int batchSizeNum = m_batchSizeNum;
    double learningRate = m_LearningRate;
    int epochNum = m_epochNum;

    printf("CreateModelFromFile(%d, %s, %s, %s, %d, %f, %d, %d, %d)\n", __LINE__, modelPath, dataPath, dataSetType,
           batchSizeNum,
           learningRate, epochNum, trainSize, testSize);

    float accuracy = 0.0;
    float tmp_loss = 0.0;

    size_t trainSamples;
    size_t testSamples;

    // load computational graph
    auto varMap = Variable::loadMap(modelPath);
    auto inputOutputs = Variable::getInputAndOutput(varMap);
    auto inputs = Variable::mapToSequence(inputOutputs.first);
    auto outputs = Variable::mapToSequence(inputOutputs.second);

    // convert to trainable module
    std::shared_ptr <Module> model(NN::extract(inputs, outputs, true));

    auto exe = Executor::getGlobalExecutor();
    BackendConfig config;
    exe->setGlobalExecutorConfig(MNN_FORWARD_CPU, config, 4);

    std::shared_ptr <SGD> sgd(new SGD(model));
    sgd->setMomentum(0.9f);
    sgd->setWeightDecay(0.0005f);

    float progress = 10.0;
    m_progress_callback(progress);
    if (bRunStopFlag) {
        printf("Training Stop By User.");
        return nullptr;
    }
    DatasetPtr dataset;
    DatasetPtr testDataset;
    VARP forwardInput;

    if (strcmp(dataSetType, "mnist") == 0) {  // mnist dataset
        printf("loading mnist\n");
        dataset = MnistDataset::create(dataPath, MnistDataset::Mode::TRAIN, (int32_t) trainSize,
                                       (int32_t) testSize);
        testDataset = MnistDataset::create(dataPath, MnistDataset::Mode::TEST, (int32_t) trainSize,
                                           (int32_t) testSize);
        forwardInput = _Input({1, 1, 28, 28}, NC4HW4);
    } else if (strcmp(dataSetType, "cifar10") == 0) { // cifar10 dataset
        printf("loading cifar10\n");
        dataset = Cifar10Dataset::create(dataPath, Cifar10Dataset::Mode::TRAIN, (int32_t) trainSize,
                                         (int32_t) testSize);
        testDataset = Cifar10Dataset::create(dataPath, Cifar10Dataset::Mode::TEST,
                                             (int32_t) trainSize, (int32_t) testSize);
        forwardInput = _Input({1, 3, 32, 32}, NC4HW4);
    }
    progress = 20.0;
    m_progress_callback(progress);
    printf("dataloader done\n");

    if (bRunStopFlag) {
        printf("Training Stop By User.");
        return nullptr;
    }
    const auto batchSize = (size_t) batchSizeNum;
    const size_t numWorkers = 0;
    bool shuffle = true;
    auto dataLoader = std::shared_ptr<DataLoader>(
            dataset.createLoader(batchSize, true, shuffle, numWorkers));
    size_t iterations = dataLoader->iterNumber();
    trainSamples = dataLoader->size();

    if (bRunStopFlag) {
        printf("Training Stop By User.");
        return nullptr;
    }
    const size_t testBatchSize = 20;
    const size_t testNumWorkers = 0;
    shuffle = false;
    auto testDataLoader = std::shared_ptr<DataLoader>(
            testDataset.createLoader(testBatchSize, true, shuffle, testNumWorkers));
    size_t testIterations = testDataLoader->iterNumber();
    testSamples = testDataLoader->size();

    // start training
    for (unsigned int epoch = 0; epoch < (int) epochNum; ++epoch) {
        if (bRunStopFlag) {
            printf("Training Stop By User.\n");
            return nullptr;
        }
        curEpoch = epoch;
        model->clearCache();
        exe->gc(Executor::FULL);
        exe->resetProfile();
        {
            dataLoader->reset();
            model->setIsTraining(true);
            int lastIndex = 0;
            int moveBatchSize = 0;
            for (int i = 0; i < iterations; i++) {
                if (bRunStopFlag) {
                    printf("Training Stop By User.\n");
                    return nullptr;
                }
                auto trainData = dataLoader->next();
                auto example = trainData[0];
                auto cast = _Cast<float>(example.first[0]);
                example.first[0] = cast * _Const(1.0f / 255.0f);
                moveBatchSize += example.first[0]->getInfo()->dim[0];
                // Compute One-Hot
                auto newTarget = _OneHot(_Cast<int32_t>(example.second[0]), _Scalar<int>(10),
                                         _Scalar<float>(1.0f),
                                         _Scalar<float>(0.0f));
                auto predict = model->forward(example.first[0]);
                auto loss = _CrossEntropy(predict, newTarget);
                auto lossvalue = loss->readMap<float>();
                tmp_loss = *lossvalue;
                m_loss_callback(epoch, tmp_loss);
                curLoss = tmp_loss;
                progress = 20.0 + 70.0 * ((epoch * iterations) + i) / (epochNum * iterations);
                m_progress_callback(progress);
                // float rate = LrScheduler::inv(0.01, (int) (epoch * iterations + i), 0.0001, 0.75);
                auto rate = (float) learningRate;
                sgd->setLearningRate(rate);
                if (moveBatchSize % (10 * batchSize) == 0 || i == iterations - 1) {
                    printf("epoch: %d %d / %ld, loss: %f, lr: %f, / %d iter\n", epoch, moveBatchSize,
                           dataLoader->size(), loss->readMap<float>()[0], rate, (i - lastIndex));
                    lastIndex = i;
                }
                sgd->step(loss);
            }
        }

        int correct = 0;
        testDataLoader->reset();
        model->setIsTraining(false);
        int moveBatchSize = 0;
        // start testing
        for (int i = 0; i < testIterations; i++) {
            if (bRunStopFlag) {
                printf("Training Stop By User.");
                return nullptr;
            }
            auto data = testDataLoader->next();
            auto example = data[0];
            moveBatchSize += example.first[0]->getInfo()->dim[0];
            if ((i + 1) % 100 == 0) {
                printf("test: %d / %ld", moveBatchSize, testDataLoader->size());
            }
            auto cast = _Cast<float>(example.first[0]);
            example.first[0] = cast * _Const(1.0f / 255.0f);
            auto testPredict = model->forward(example.first[0]);
            testPredict = _ArgMax(testPredict, 1);
            auto accu = _Cast<int32_t>(_Equal(testPredict, _Cast<int32_t>(example.second[0]))).sum(
                    {});
            correct += accu->readMap<int32_t>()[0];
        }

        auto accu = (float) correct / (float) testSamples;
        printf("epoch: %d  accuracy: %f\n", epoch, accu);
        m_accuracy_callback(epoch, accu);
        accuracy = accu;
        exe->dumpProfile();
    }
    printf("model train done\n");
    if (bRunStopFlag) {
        printf("Training Stop By User.\n");
        return nullptr;
    }

    model->setIsTraining(true);  // save the training state computation graph
    forwardInput->setName("data");
    auto inputPredict = model->forward(forwardInput);
    inputPredict->setName("prob");
//    Transformer::turnModelToInfer()->onExecute({inputPredict});
    // progress = 95;
    // onTrainProgressCallback(env, listener, onProgressMethodID, progress);
    if (bRunStopFlag) {
        printf("Training Stop By User.\n");
        return nullptr;
    }
    Variable::save({inputPredict}, modelPath);
    printf("model save done\n");
    // progress = 100;
    // onTrainProgressCallback(env, listener, onProgressMethodID, progress);
    std::string result = std::to_string(tmp_loss) + "," + std::to_string(trainSamples) +
                         "," + std::to_string(accuracy) + "," + std::to_string(testSamples);
    // env->ReleaseStringUTFChars(modelCachePath, modelPath);
    // env->ReleaseStringUTFChars(dataCachePath, dataPath);
    // env->ReleaseStringUTFChars(dataSet, dataSetType);

    // return env->NewStringUTF(result.data());
    return result;
}