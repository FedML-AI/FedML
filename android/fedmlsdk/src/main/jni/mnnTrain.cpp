#ifdef FEDML_ANDROID_MNNTRAIN_CPP
#define FEDML_ANDROID_MNNTRAIN_CPP

#include "mnnTrain.h"
#include "jniAssist.h"
#include "JavaBundle.h"

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Train;

// Mobile NN
// Global variable to get the current epoch value and current loss in this epoch
static jint curEpoch = 0;
static jfloat curLoss = 0.0;

static bool bRunStopFlag = false;

extern "C"
void
onTrainCallback(JNIEnv *env, jobject subscriber, jmethodID methodId, jint epoch, jfloat value) {
    env->CallVoidMethod(subscriber, methodId, epoch, value);
}

extern "C"
void
onTrainProgressCallback(JNIEnv *env, jobject subscriber, jmethodID methodId, jint progress) {
    env->CallVoidMethod(subscriber, methodId, progress);
}

extern "C"
JNIEXPORT jstring JNICALL
Java_ai_fedml_edge_nativemnn_NativeMnn_getEpochAndLoss(JNIEnv *env, jclass) {
    std::string result = std::to_string(curEpoch) + "," + std::to_string(curLoss);
    return env->NewStringUTF(result.data());
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_ai_fedml_edge_nativemnn_NativeMnn_stopTraining(JNIEnv *, jclass) {
    bRunStopFlag = true;
    LOGD("stopTraining By User.");
    return JNI_TRUE;
}


// @Return: trainSamples, testSamples, LOSS, ACC
extern "C"
JNIEXPORT jstring JNICALL
Java_ai_fedml_edge_nativemnn_NativeMnn_train(JNIEnv *env, jclass,
                                             jstring modelCachePath, jstring dataCachePath,
                                             jstring dataSet,
                                             jint trainSize, jint testSize,
                                             jint batchSizeNum, jdouble learningRate, jint epochNum,
                                             jobject listener) {
    bRunStopFlag = false;
    const char *modelPath = env->GetStringUTFChars(modelCachePath, nullptr);
    const char *dataPath = env->GetStringUTFChars(dataCachePath, nullptr);
    const char *dataSetType = env->GetStringUTFChars(dataSet, nullptr);

    LOGD("CreateModelFromFile(%s, %s, %s, %d, %f, %d, %d, %d)", modelPath, dataPath, dataSetType, batchSizeNum,
         learningRate, epochNum, trainSize, testSize);
    jmethodID onLossMethodID = getMethodIdByNameAndSig(env, listener, "onEpochLoss", "(IF)V");
    jmethodID onAccuracyMethodID = getMethodIdByNameAndSig(env, listener, "onEpochAccuracy",
                                                           "(IF)V");
    jmethodID onProgressMethodID = getMethodIdByNameAndSig(env, listener, "onProgressChanged",
                                                           "(I)V");
    int progress = 0;
    float accuracy = 0.0;
    float tmp_loss = 0.0;
    size_t trainSamples;
    size_t testSamples;
    onTrainProgressCallback(env, listener, onProgressMethodID, progress);
    onTrainCallback(env, listener, onAccuracyMethodID, 0, (jfloat) accuracy);
    onTrainCallback(env, listener, onLossMethodID, 0, (jfloat) tmp_loss);

    // load computational graph
    auto varMap = Variable::loadMap(modelPath);
    auto inputOutputs = Variable::getInputAndOutput(varMap);
    auto inputs = Variable::mapToSequence(inputOutputs.first);
    auto outputs = Variable::mapToSequence(inputOutputs.second);

    // convert to trainable module
    std::shared_ptr<Module> model(NN::extract(inputs, outputs, true));

    // 配置训练框架参数
    auto exe = Executor::getGlobalExecutor();
    BackendConfig config;
    // 使用CPU，4线程
    exe->setGlobalExecutorConfig(MNN_FORWARD_CPU, config, 4);

    // SGD求解器并设置参数
    std::shared_ptr<SGD> sgd(new SGD(model));
    sgd->setMomentum(0.9f);
    sgd->setWeightDecay(0.0005f);
    progress = 10;
    onTrainProgressCallback(env, listener, onProgressMethodID, progress);
    if (bRunStopFlag) {
        LOGD("Training Stop By User.");
        return nullptr;
    }
    // 创建数据集和DataLoader
    DatasetPtr dataset;
    DatasetPtr testDataset;
    VARP forwardInput;
    if (strcmp(dataSetType, "mnist") == 0) {  // mnist dataset
        LOGD("loading mnist");
        dataset = MnistDataset::create(dataPath, MnistDataset::Mode::TRAIN, (int32_t) trainSize,
                                       (int32_t) testSize);
        testDataset = MnistDataset::create(dataPath, MnistDataset::Mode::TEST, (int32_t) trainSize,
                                           (int32_t) testSize);
        forwardInput = _Input({1, 1, 28, 28}, NC4HW4);
    } else if (strcmp(dataSetType, "cifar10") == 0) { // cifar10 dataset
        LOGD("loading cifar10");
        dataset = Cifar10Dataset::create(dataPath, Cifar10Dataset::Mode::TRAIN, (int32_t) trainSize,
                                         (int32_t) testSize);
        testDataset = Cifar10Dataset::create(dataPath, Cifar10Dataset::Mode::TEST,
                                             (int32_t) trainSize, (int32_t) testSize);
        forwardInput = _Input({1, 3, 32, 32}, NC4HW4);
    }
    progress = 20;
    onTrainProgressCallback(env, listener, onProgressMethodID, progress);
    LOGD("dataloader done");

    if (bRunStopFlag) {
        LOGD("Training Stop By User.");
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
        LOGD("Training Stop By User.");
        return nullptr;
    }
    // test data setting
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
            LOGD("Training Stop By User.");
            return nullptr;
        }
        curEpoch = (jint) epoch;
        model->clearCache();
        exe->gc(Executor::FULL);
        exe->resetProfile();
        {
            dataLoader->reset();
            // 训练阶段需设置isTraining Flag为true
            model->setIsTraining(true);
            int lastIndex = 0;
            int moveBatchSize = 0;
            for (int i = 0; i < iterations; i++) {
                if (bRunStopFlag) {
                    LOGD("Training Stop By User.");
                    return nullptr;
                }
                // 获得一个batch的数据，包括数据及其label
                auto trainData = dataLoader->next();
                auto example = trainData[0];
                auto cast = _Cast<float>(example.first[0]);
                example.first[0] = cast * _Const(1.0f / 255.0f);
                moveBatchSize += example.first[0]->getInfo()->dim[0];
                // Compute One-Hot
                auto newTarget = _OneHot(_Cast<int32_t>(example.second[0]), _Scalar<int>(10),
                                         _Scalar<float>(1.0f),
                                         _Scalar<float>(0.0f));
                // 前向计算
                auto predict = model->forward(example.first[0]);
                // 计算loss
                auto loss = _CrossEntropy(predict, newTarget);
                auto lossvalue = loss->readMap<float>();
                tmp_loss = *lossvalue;
                // 回调
                onTrainCallback(env, listener, onLossMethodID, (jint) epoch, tmp_loss);
                curLoss = tmp_loss;
                // 进度计算
                progress = 20 + 70 * ((epoch * iterations) + i) / (epochNum * iterations);
                onTrainProgressCallback(env, listener, onProgressMethodID, progress);
                // 调整学习率
                // float rate = LrScheduler::inv(0.01, (int) (epoch * iterations + i), 0.0001, 0.75);
                auto rate = (float) learningRate;
                sgd->setLearningRate(rate);
                if (moveBatchSize % (10 * batchSize) == 0 || i == iterations - 1) {
                    LOGD("epoch: %d %d / %ld, loss: %f, lr: %f, / %d iter", epoch, moveBatchSize,
                         dataLoader->size(), loss->readMap<float>()[0], rate, (i - lastIndex));
                    lastIndex = i;
                }
                // 根据loss反向计算，并更新网络参数
                sgd->step(loss);
            }
        }

        // 测试模型
        int correct = 0;
        testDataLoader->reset();
        // 测试时，需设置标志位
        model->setIsTraining(false);
        int moveBatchSize = 0;
        // start testing
        for (int i = 0; i < testIterations; i++) {
            if (bRunStopFlag) {
                LOGD("Training Stop By User.");
                return nullptr;
            }
            auto data = testDataLoader->next();
            auto example = data[0];
            moveBatchSize += example.first[0]->getInfo()->dim[0];
            if ((i + 1) % 100 == 0) {
                LOGD("test: %d / %ld", moveBatchSize, testDataLoader->size());
            }
            auto cast = _Cast<float>(example.first[0]);
            example.first[0] = cast * _Const(1.0f / 255.0f);
            auto testPredict = model->forward(example.first[0]);
            testPredict = _ArgMax(testPredict, 1);
            auto accu = _Cast<int32_t>(_Equal(testPredict, _Cast<int32_t>(example.second[0]))).sum(
                    {});
            correct += accu->readMap<int32_t>()[0];
        }

        // 计算准确率
        auto accu = (float) correct / (float) testSamples;
        LOGD("epoch: %d  accuracy: %f\n", epoch, accu);
        onTrainCallback(env, listener, onAccuracyMethodID, epoch, accu);
        accuracy = accu;
        exe->dumpProfile();
    }
    LOGD("model train done");
    if (bRunStopFlag) {
        LOGD("Training Stop By User.");
        return nullptr;
    }

    model->setIsTraining(true);  // save the training state computation graph
    forwardInput->setName("data");
    auto inputPredict = model->forward(forwardInput);
    inputPredict->setName("prob");
//    Transformer::turnModelToInfer()->onExecute({inputPredict});
    progress = 95;
    onTrainProgressCallback(env, listener, onProgressMethodID, progress);
    // 保存输出节点，会连同结构参数一并存储下来
    if (bRunStopFlag) {
        LOGD("Training Stop By User.");
        return nullptr;
    }
    Variable::save({inputPredict}, modelPath);
    LOGD("model save done");
    progress = 100;
    onTrainProgressCallback(env, listener, onProgressMethodID, progress);
    std::string result = std::to_string(tmp_loss) + "," + std::to_string(trainSamples) +
                         "," + std::to_string(accuracy) + "," + std::to_string(testSamples);
    env->ReleaseStringUTFChars(modelCachePath, modelPath);
    env->ReleaseStringUTFChars(dataCachePath, dataPath);
    env->ReleaseStringUTFChars(dataSet, dataSetType);

    return env->NewStringUTF(result.data());
}
#endif //FEDML_ANDROID_MNNTRAIN_CPP
