#include <torch/torch.h>

#include <string>

namespace torch {
namespace data {
namespace datasets {
class CIFAR10 : public torch::data::datasets::Dataset<CIFAR10> {
 public:
  // The mode in which the dataset is loaded.
  enum class Mode { kTrain, kTest };

  explicit CIFAR10(const std::string &root, uint32_t kSize, Mode mode = Mode::kTrain);

  // https://pytorch.org/cppdocs/api/structtorch_1_1data_1_1_example.html#structtorch_1_1data_1_1_example
  torch::data::Example<> get(size_t index) override;

  torch::optional<size_t> size() const override;

  bool is_train() const noexcept;

  // Returns all images stacked into a single tensor.
  const torch::Tensor &images() const;

  const torch::Tensor &targets() const;

 private:
  // Returns all targets stacked into a single tensor.
  torch::Tensor images_, targets_;
};
}  // namespace datasets
}  // namespace data
}  // namespace torch