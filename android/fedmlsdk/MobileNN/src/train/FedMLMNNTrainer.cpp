#include "FedMLMNNTrainer.h"

std::string FedMLMNNTrainer::train() {
    const char* modelCachePath = m_modelCachePath.c_str();
    const char* dataCachePath = m_dataCachePath.c_str();
    const char* dataSet = m_dataSet.c_str();

    // load model
    auto varMap = Variable::loadMap(modelCachePath);
    auto inputOutputs = Variable::getInputAndOutput(varMap);
    auto inputs = Variable::mapToSequence(inputOutputs.first);
    auto outputs = Variable::mapToSequence(inputOutputs.second);

    std::shared_ptr <Module> model(NN::extract(inputs, outputs, true));

    // set executor
    auto exe = Executor::getGlobalExecutor();
    BackendConfig config;
    exe->setGlobalExecutorConfig(MNN_FORWARD_CPU, config, 4);

    // set optimizer
    std::shared_ptr <SGD> sgd(new SGD(model));
    sgd->setLearningRate(m_LearningRate);
    sgd->setMomentum(0.1f);

    m_progress_callback(10.0f);
    if (bRunStopFlag) {printf("Training Stop By User."); return nullptr;}

    // load data
    DatasetPtr dataset;
    DatasetPtr testDataset;
    VARP forwardInput;

    if (strcmp(dataSet, "mnist") == 0) {
        printf("loading mnist\n");
        dataset = MnistDataset::create(dataCachePath, MnistDataset::Mode::TRAIN, m_trainSize, m_testSize);
        testDataset = MnistDataset::create(dataCachePath, MnistDataset::Mode::TEST, m_trainSize, m_testSize);
        forwardInput = _Input({1, 1, 28, 28}, NC4HW4);
    } else if (strcmp(dataSet, "cifar10") == 0) {
        printf("loading cifar10\n");
        dataset = Cifar10Dataset::create(dataCachePath, Cifar10Dataset::Mode::TRAIN, m_trainSize, m_testSize);
        testDataset = Cifar10Dataset::create(dataCachePath, Cifar10Dataset::Mode::TEST, m_trainSize, m_testSize);
        forwardInput = _Input({1, 3, 32, 32}, NC4HW4);
    }
    auto dataLoader = std::shared_ptr<DataLoader>(dataset.createLoader(m_batchSizeNum, true, true, 0));
    size_t iterations = dataLoader->iterNumber();
    size_t trainSamples = dataLoader->size();

    auto testDataLoader = std::shared_ptr<DataLoader>(testDataset.createLoader(m_batchSizeNum, true, false, 0));
    size_t testIterations = testDataLoader->iterNumber();
    size_t testSamples = testDataLoader->size();

    m_progress_callback(20.0f);
    if (bRunStopFlag) {printf("Training Stop By User."); return nullptr;}

    // model training
    for (int epoch = 0; epoch < m_epochNum; ++epoch) {
        curEpoch = epoch;

        model->clearCache();
        exe->gc(Executor::FULL);
        exe->resetProfile();
        {
            dataLoader->reset();
            model->setIsTraining(true);

            int moveBatchSize = 0;
            for (int i = 0; i < iterations; i++) {
                if (bRunStopFlag) {printf("Training Stop By User."); return nullptr;}

                auto trainData  = dataLoader->next();
                auto example    = trainData[0];
                auto cast       = _Cast<float>(example.first[0]);
                example.first[0] = cast * _Const(1.0f / 255.0f);
                moveBatchSize += example.first[0]->getInfo()->dim[0];

                // Compute One-Hot
                auto newTarget = _OneHot(_Cast<int32_t>(example.second[0]), _Scalar<int>(10), _Scalar<float>(1.0f),
                                         _Scalar<float>(0.0f));

                auto predict = model->forward(example.first[0]);
                auto loss    = _CrossEntropy(predict, newTarget);
                sgd->step(loss);

                curLoss = loss->readMap<float>()[0];
                m_loss_callback(epoch, loss->readMap<float>()[0]);
                m_progress_callback(20.0 + 70.0*((epoch*iterations)+i)/(m_epochNum*iterations));

                if (moveBatchSize % (10 * m_batchSizeNum) == 0 || i == iterations - 1) {
                    std::cout << "epoch: " << (epoch);
                    std::cout << "  " << moveBatchSize << " / " << dataLoader->size();
                    std::cout << " loss: " << loss->readMap<float>()[0] << std::endl;
                    std::cout.flush();
                }
            }
        }
        // model testing
        int correct = 0;
        testDataLoader->reset();
        model->setIsTraining(false);
        int moveBatchSize = 0;
        for (int i = 0; i < testIterations; i++) {
            if (bRunStopFlag) {printf("Training Stop By User."); return nullptr;}

            auto data = testDataLoader->next();
            auto example = data[0];
            moveBatchSize += example.first[0]->getInfo()->dim[0];
            if ((i + 1) % 100 == 0) {
                std::cout << "test: " << moveBatchSize << " / " << testDataLoader->size() << std::endl;
            }
            auto cast = _Cast<float>(example.first[0]);
            example.first[0] = cast * _Const(1.0f / 255.0f);
            auto testPredict = model->forward(example.first[0]);
            testPredict = _ArgMax(testPredict, 1);
            auto accu = _Cast<int32_t>(_Equal(testPredict, _Cast<int32_t>(example.second[0]))).sum(
                    {});
            correct += accu->readMap<int32_t>()[0];
        }

        // get accuracy
        auto accu = (float) correct / (float) testSamples;
        std::cout << "epoch: " << epoch << "  accuracy: " << accu << std::endl;
        m_accuracy_callback(epoch, accu);
        exe->dumpProfile();
    }

    // model saving
    model->setIsTraining(true);  // save the training state computation graph
    forwardInput->setName("data");
    auto inputPredict = model->forward(forwardInput);
    inputPredict->setName("prob");
    Variable::save({inputPredict}, modelCachePath);
    printf("model save done\n");

    std::string result = std::to_string(trainSamples);
    return result;
}

