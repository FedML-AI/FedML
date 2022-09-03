#include "FedMLTorchTrainer.h"

using namespace torch;
using namespace torch::jit;

std::string FedMLTorchTrainer::train() {
    const char* modelCachePath = m_modelCachePath.c_str();
    const char* dataCachePath = m_dataCachePath.c_str();
    const char* dataSet = m_dataSet.c_str();

    mobile::Module model = _load_for_mobile(modelCachePath);

    std::vector<at::Tensor> parameters = model.parameters();

    double momentum = 0.1;
    mobile::SGD optimizer(parameters, mobile::SGDOptions(m_LearningRate).momentum(momentum));

    auto train_dataset =
            MNIST(dataCachePath, m_trainSize)
                    .map(data::transforms::Normalize<>(0.1307, 0.3081))
                    .map(data::transforms::Stack<>());
    int train_dataset_size = train_dataset.size().value();
    auto train_loader =
            data::make_data_loader<mobile::SequentialSampler>(
                    std::move(train_dataset), m_batchSizeNum);

    auto test_dataset =
            MNIST(dataCachePath, m_testSize, MNIST::Mode::kTest)
                    .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                    .map(torch::data::transforms::Stack<>());
    int test_dataset_size = test_dataset.size().value();
    auto test_loader =
            data::make_data_loader<mobile::SequentialSampler>(
                    std::move(test_dataset), m_batchSizeNum);

    for (int epoch = 0; epoch < m_epochNum; ++epoch) {
        model.train();
        int batch_idx = 0;
        for (const auto& batch : *train_loader) {
            auto inputs = batch.data;
            auto labels = batch.target;
            optimizer.zero_grad();
            std::vector<c10::IValue> train_inputs{inputs};
            auto outputs = model.forward(train_inputs).toTensor();
            auto loss = nll_loss(outputs, labels);
            loss.backward();
            optimizer.step();

            if (batch_idx++ % 10 == 0) {
                printf("Train Epoch: %d [%5ld/%5d] Loss: %.4f\n", epoch,
                       batch_idx * inputs.size(0), train_dataset_size,
                       loss.template item<float>());
            }
        }
    }

    model.eval();
    int64_t correct = 0;
    for (const auto& batch : *test_loader) {
        auto inputs = batch.data;
        auto labels = batch.target;
        std::vector<c10::IValue> train_inputs{inputs};
        auto outputs = model.forward(train_inputs).toTensor();
        auto preds = outputs.argmax(1);
        correct += torch::sum(preds == labels).template item<int64_t>();
    }
    printf("Accuracy: %.3f\n", static_cast<double>(correct) / test_dataset_size);

    _save_parameters(model.named_parameters(), "model_param_trained.pt");

    return "pytorch train done";
}