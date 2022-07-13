# Examples

- This is the outline of all examples. For more detailed instructions, please refer to [https://doc.fedml.ai](https://doc.fedml.ai)
- In [FedML/python/app](./../app) folder, we also provide applications in real-world settings.

## Cross-silo Federated Learning for cross-organization/account training

using communication backend MQTT_S3: `python/examples/cross_silo/mqtt_s3_fedavg_mnist_lr_example`

using communication backend gRPC: `python/examples/cross_silo/grpc_fedavg_mnist_lr_example`

using communication backend MPI: `python/examples/cross_silo/mpi_fedavg_mnist_lr_example`

using communication backend PyTorch RPC: `python/examples/cross_silo/trpc_fedavg_mnist_lr_example`

using communication backend CUDA RPC: `python/examples/cross_silo/cuda_rpc_fedavg_mnist_lr_example`

hierarchical cross-silo federated learning: `python/examples/cross_silo/mqtt_s3_fedavg_hierarchical_mnist_lr_example` 

## Cross-device Federated Learning for Smartphones

using communication backend MQTT_S3_MNN (tested): `python/examples/cross_device/mqtt_s3_fedavg_mnist_lr_example`


## Simulation: (1) simulate FL using a single process (2) MPI-based FL Simulator (3) NCCL-based FL Simulator (fastest)

### (1) Single Process (standalone)
- Decentralized FL on MNIST dataset with Logistic Regression model (tested): `python/examples/simulation/sp_decentralized_mnist_lr_example`

- FedAvg on Cifar10 dataset with CNN model (tested): 

  `python/examples/simulation/sp_fedavg_cifar10_cnn_example`
  
- FedAvg on Cifar10 dataset with MobileNet model (tested): 

  `python/examples/simulation/sp_fedavg_cifar10_mobilenet_example`

* FedAvg on Cifar10 dataset with ResNet56 model (tested): `python/examples/simulation/sp_fedavg_cifar10_resnet56_example`

- FedAvg on MNIST dataset with Logistic Regression model (tested): `python/examples/simulation/sp_fedavg_mnist_lr_example`
- FedAvg on Stackoverflow_lr dataset with Logistic Regression model (tested): `python/examples/simulation/sp_fedavg_stackoverflow_lr_lr_example`
- FedNova on MNIST dataset with Logistic Regression model (tested): `python/examples/simulation/sp_fednova_mnist_lr_example`
- FedOpt on Cifar10 dataset with  MobileNetV3 model (tested): `python/examples/simulation/sp_fednova_mnist_lr_example`
- FedOpt on MNIST dataset with Logistic Regression model (tested): `python/examples/simulation/sp_fedopt_mnist_lr_example`
- FedSGD on Cifar10 dataset with ResNet20 model (tested): `python/examples/simulation/sp_fedsgd_cifar10_resnet20_example`
- HierarchicalFL on MNIST dataset with Logistic Regression model (tested): `python/examples/simulation/sp_hierarchicalfl_mnist_lr_example`
- Turboaggregate on MNIST dataset with Logistic Regression model (tested): `python/examples/simulation/sp_turboaggregate_mnist_lr_example`
- VerticalFL  on MNIST dataset with Logistic Regression model (tested):  `python/examples/simulation/sp_vertical_mnist_lr_example`

### (2) MPI-based Simulator (distributed)

- Base framework on MNIST dataset with Logistic Regression model (tested): `python/examples/simulation/mpi_base_framework_example`
- Decentralized FL on MNIST dataset with Logistic Regression model (tested): `python/examples/simulation/mpi_decentralized_fl_example`

- FedAvg on different datasets with different models:

  `python/examples/simulation/mpi_fedavg_datasets_and_models_example`,includes:

  - Dataset: Cifar10; Model: MobileNet(tested)
  - Dataset: Cifar100; Model: MobileNetV3(tested)
  - Dataset: Cinic Model: Resnet56(tested)
  - Dataset: Fedcifar100; Model: Resnet18(tested)
  - Dataset: Fedemnist; Model: CNN(tested)
  - Dataset: Fedshakespeare; Model: RNN(tested)
  - Dataset: lending_club_loan; Model: Logistic Regression(tested)
  - Dataset: MNIST; Model: CNN(tested)
  - Dataset: MNIST; Model: Logistic Regression(tested)
  - Dataset: Shakespeare; Model: RNN(tested)
  - Dataset: Stackoverflow_lr; Model: Logistic Regression
  - Dataset: Stackoverflow_nwp; Model: RNN

- Robust FedAvg on Cifar10 dataset with Resnet56 (tested):

  `python/examples/simulation/mpi_fedavg_robust_example`

- FedOpt on different datasets with different models:

  `python/examples/simulation/mpi_fedopt_datasets_and_models_example`,includes:

  - Dataset: Cifar10; Model: MobileNet(tested)
  - Dataset: Cifar100; Model: MobileNetV3(tested)
  - Dataset: Cinic Model: Resnet56(tested)
  - Dataset: Fedcifar100; Model: Resnet18(tested)
  - Dataset: Fedemnist; Model: CNN(tested)
  - Dataset: Fedshakespeare; Model: RNN(tested)
  - Dataset: lending_club_loan; Model: Logistic Regression(tested)
  - Dataset: MNIST; Model: CNN(tested)
  - Dataset: MNIST; Model: Logistic Regression(tested)
  - Dataset: Shakespeare; Model: RNN(tested)
  - Dataset: Stackoverflow_lr; Model: Logistic Regression
  - Dataset: Stackoverflow_nwp; Model: RNN

- FedProx on different datasets with different models:

  `python/examples/simulation/mpi_fedprox_datasets_and_models_example`,includes:

  - Dataset: Cifar10; Model: MobileNet(tested)
  - Dataset: Cifar100; Model: MobileNetV3(tested)
  - Dataset: Cinic Model: Resnet56(tested)
  - Dataset: Fedcifar100; Model: Resnet18(tested)
  - Dataset: Fedemnist; Model: CNN(tested)
  - Dataset: Fedshakespeare; Model: RNN(tested)
  - Dataset: lending_club_loan; Model: Logistic Regression(tested)
  - Dataset: MNIST; Model: CNN(tested)
  - Dataset: MNIST; Model: Logistic Regression(tested)
  - Dataset: Shakespeare; Model: RNN(tested)
  - Dataset: Stackoverflow_lr; Model: Logistic Regression
  - Dataset: Stackoverflow_nwp; Model: RNN

- FedAvg on MNIST dataset with Logistic Regression model (tested): `python/examples/simulation/mpi_torch_fedavg_mnist_lr_example`

- FedGAN on MNIST dataset with Generating adversarial networks (tested): `python/examples/simulation/mpi_torch_fedgan_mnist_gan_example`

- FedGKT on MNIST dataset with Logistic Regression model (tested): `python/examples/simulation/mpi_torch_fedgkt_mnist_lr_example`

- FedNAS on Cifar10 dataset with DART model (tested): `python/examples/simulation/mpi_torch_fednas_cifar10_dart_example`

- FedOpt on MNIST dataset with Logistic Regression model (tested): `python/examples/simulation/mpi_torch_fedopt_mnist_lr_example`

- FedProx on MNIST dataset with Logistic Regression model (tested): `python/examples/simulation/mpi_torch_fedprox_mnist_lr_example`

- SplitNN on Cifar10 dataset with ResNet56 model (tested): `python/examples/simulation/mpi_torch_splitnn_cifar10_resnet56_example`

### (3) NCCL-based Simulator 

* FedAvg on Cifar10 dataset with ResNet56 model: `python/examples/simulation/nccl_fedavg_example`

## Distributed Training: Accelerate Model Training with Lightweight Cheetah

Coming soon