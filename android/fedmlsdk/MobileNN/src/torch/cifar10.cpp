// Mimicing
// https://github.com/pytorch/pytorch/blob/fc804b5def5e7d7ecad24c4d1ca4ac575e588ae8/torch/csrc/api/src/data/datasets/mnist.cpp

// CIFAR dataset
// https://www.cs.toronto.edu/~kriz/cifar.html

// #include <bits/stdint-uintn.h>
#include "cifar10.h"

#include <torch/torch.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace torch {
namespace data {
namespace datasets {
const std::vector<std::string> train_set_file_names{
    "data_batch_1.bin"};
const std::vector<std::string> test_set_file_names{"test_batch.bin"};
const std::string meta_data_file_name{"batches.meta.txt"};

constexpr const uint32_t num_samples_per_file{10000};
constexpr const uint32_t image_height{32};
constexpr const uint32_t image_width{32};
constexpr const uint32_t image_channels{3};

std::string join_paths(std::string head, const std::string& tail) {
  if (head.back() != '/') {
    head.push_back('/');
  }
  head += tail;
  return head;
}

torch::Tensor read_targets_from_file(const std::string& file_path, uint32_t kSize) {
  torch::Tensor targets =
      torch::empty({kSize * 1}, torch::kUInt8);
  uint8_t* ptr_data = targets.data_ptr<uint8_t>();

  std::fstream f;
  f.open(file_path, f.binary | f.in);
  if (!f.is_open()) {
    std::cerr << "Failed to open " << file_path << std::endl;
    // TORCH_CHECK(f, "Error opening targets file at ", file_path);
  } else {
    for(uint32_t i=0; i<kSize; i++) {
      f.read(reinterpret_cast<char*>(ptr_data + i * 1), 1);
      f.ignore(image_height * image_width * image_channels * 1);
    }
  }

  // assert(
  //     (count == num_samples_per_file) &&
  //     "Insufficient number of targets. Data file might have been corrupted.");

  // targets = targets.reshape({num_samples_per_file, 1});

  return targets;
}

torch::Tensor read_images_from_file(const std::string& file_path, uint32_t kSize) {
  constexpr const uint32_t num_image_bytes{image_height * image_width *
                                           image_channels * 1};

  torch::Tensor images =
      torch::empty({kSize * num_image_bytes}, torch::kUInt8);
  uint8_t* ptr_data = images.data_ptr<uint8_t>();

  std::fstream f;
  f.open(file_path, f.binary | f.in);
  if (!f.is_open()) {
    std::cerr << "Failed to open " << file_path << std::endl;
    // TORCH_CHECK(f, "Error opening images file at ", file_path);
  } else {
    for(uint32_t i=0; i<kSize; i++) {
      f.ignore(1);
      f.read(reinterpret_cast<char*>(ptr_data + i * num_image_bytes),
             num_image_bytes);
    }
  }

  // assert((count == num_samples_per_file) &&
  //        "Insufficient number of images. Data file might have been corrupted.");

  // The next 3072 bytes are the values of the pixels of the image.
  // The first 1024 bytes are the red channel values, the next 1024 the green,
  // and the final 1024 the blue. The values are stored in row-major order, so
  // the first 32 bytes are the red channel values of the first row of the
  // image. NCHW format
  images = images.reshape(
      {kSize, image_channels, image_height, image_width});

  return images;
}

torch::Tensor read_images(const std::string& root, uint32_t kSize, bool train) {
  std::vector<std::string> data_set_file_names;
  if (train) {
    data_set_file_names = train_set_file_names;
  } else {
    data_set_file_names = test_set_file_names;
  }

  std::vector<std::string> data_set_file_paths;
  for (const std::string& data_set_file_name : data_set_file_names) {
    data_set_file_paths.push_back(join_paths(root, data_set_file_name));
  }

  std::vector<torch::Tensor> image_tensors;

  for (const std::string& data_set_file_path : data_set_file_paths) {
    torch::Tensor images = read_images_from_file(data_set_file_path, kSize);
    image_tensors.push_back(images);
  }

  torch::Tensor images = torch::cat(image_tensors, 0);

  images = images.to(torch::kFloat32).div_(255);

  return images;
}

torch::Tensor read_targets(const std::string& root, uint32_t kSize, bool train) {
  std::vector<std::string> data_set_file_names;
  if (train) {
    data_set_file_names = train_set_file_names;
  } else {
    data_set_file_names = test_set_file_names;
  }

  std::vector<std::string> data_set_file_paths;
  for (const std::string& data_set_file_name : data_set_file_names) {
    data_set_file_paths.push_back(join_paths(root, data_set_file_name));
  }

  std::vector<torch::Tensor> target_tensors;

  for (const std::string& data_set_file_path : data_set_file_paths) {
    torch::Tensor targets = read_targets_from_file(data_set_file_path, kSize);
    target_tensors.push_back(targets);
  }

  torch::Tensor targets = torch::cat(target_tensors, 0);

  targets = targets.to(torch::kInt64);

  return targets;
}

CIFAR10::CIFAR10(const std::string& root, uint32_t kSize, Mode mode)
    : images_(read_images(root, kSize, mode == Mode::kTrain)),
      targets_(read_targets(root, kSize, mode == Mode::kTrain)) {}

torch::data::Example<> CIFAR10::get(size_t index) {
  return {images_[index], targets_[index]};
}

torch::optional<size_t> CIFAR10::size() const { return images_.size(0); }

// bool CIFAR10::is_train() const noexcept {
//   return images_.size(0) == num_samples_per_file * train_set_file_names.size();
// }

const torch::Tensor& CIFAR10::images() const { return images_; }

const torch::Tensor& CIFAR10::targets() const { return targets_; }

}  // namespace datasets
}  // namespace data
}  // namespace torch