# Examples

- This is the outline of all examples. For more detailed instructions, please refer to [https://doc.fedml.ai](https://doc.fedml.ai)
- In [FedML/python/app](./../app) folder, we also provide applications in real-world settings.

|                                            | platform/scenario     | federated optimizer | dataset           | model                           | communication backend | source code                                                  | example doc                                                  |
| ------------------------------------------ | --------------------- | ------------------- | ----------------- | ------------------------------- | --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| mqtt_s3_fedavg_mnist_lr_example            | Beehive(Cross-silo)   | FedAvg              | MNIST             | Logistic Regression             | MQTT_S3               | [Link](cross_silo/mqtt_s3_fedavg_mnist_lr_example)           | [Link](cross_silo/mqtt_s3_fedavg_mnist_lr_example/README.md) |
| grpc_fedavg_mnist_lr_example               | Beehive(Cross-silo)   | FedAvg              | MNIST             | Logistic Regression             | GRPC                  | [Link](cross_silo/grpc_fedavg_mnist_lr_example)              | [Link](cross_silo/grpc_fedavg_mnist_lr_example/README.md)    |
| mpi_fedavg_mnist_lr_example                | Beehive(Cross-silo)   | FedAvg              | MNIST             | Logistic Regression             | MPI                   | [Link](cross_silo/mpi_fedavg_mnist_lr_example)               | [Link](cross_silo/mpi_fedavg_mnist_lr_example/README.md)     |
| trpc_fedavg_mnist_lr_example               | Beehive(Cross-silo)   | FedAvg              | MNIST             | Logistic Regression             | TRPC                  | [Link](cross_silo/trpc_fedavg_mnist_lr_example)              | [Link](cross_silo/trpc_fedavg_mnist_lr_example/README.md)    |
| mqtt_s3_fedavg_mnist_lr_example            | Beehive(Cross-silo)   | FedAvg              | MNIST             | Logistic Regression             | CUDA RPC              | [Link](cross_silo/cuda_rpc_fedavg_mnist_lr_example)          | [Link](cross_silo/cuda_rpc_fedavg_mnist_lr_example/README.md) |
| grpc_fedavg_mnist_lr_example               | Beehive(Cross-silo)   | FedAvg              | MNIST             | Logistic Regression             | MQTT_S3               | [Link](cross_silo/mqtt_s3_fedavg_hierarchical_mnist_lr_example) | [Link](cross_silo/mqtt_s3_fedavg_hierarchical_mnist_lr_example/README.md) |
| mqtt_s3_fedavg_mnist_lr_example            | Beehive(Cross-device) | FedAvg              | MNIST             | Logistic Regression             | MQTT_S3               | [Link](cross_device/mqtt_s3_fedavg_mnist_lr_example)         | [Link](cross_device/mqtt_s3_fedavg_mnist_lr_example/README.md) |
| sp_decentralized_mnist_lr_example          | Parrot                | DecentralizedFL     | MNIST             | Logistic Regression             | single process        | [Link](simulation/sp_decentralized_mnist_lr_example)         | [Link](simulation/sp_decentralized_mnist_lr_example/README.md) |
| sp_fedavg_cifar10_cnn_example              | Parrot                | FedAvg              | Cifar10           | CNN                             | single process        | [Link](simulation/sp_fedavg_cifar10_cnn_example)             | [Link](simulation/sp_fedavg_cifar10_cnn_example/README.md)   |
| sp_fedavg_cifar10_mobilenet_example        | Parrot                | FedAvg              | Cifar10           | MobileNet                       | single process        | [Link](simulation/sp_fedavg_cifar10_mobilenet_example)       | [Link](simulation/sp_fedavg_cifar10_mobilenet_example/README.md) |
| sp_fedavg_cifar10_resnet56_example         | Parrot                | FedAvg              | Cifar10           | Resnet56                        | single process        | [Link](simulation/sp_fedavg_cifar10_resnet56_example)        | [Link](simulation/sp_fedavg_cifar10_resnet56_example/README.md) |
| sp_fedavg_mnist_lr_example                 | Parrot                | FedAvg              | MNIST             | Logistic Regression             | single process        | [Link](simulation/sp_fedavg_mnist_lr_example)                | [Link](simulation/sp_fedavg_mnist_lr_example/README.md)      |
| sp_fedavg_stackoverflow_lr_lr_example      | Parrot                | FedAvg              | Stackoverflow_lr  | Logistic Regression             | single process        | [Link](simulation/sp_fedavg_stackoverflow_lr_lr_example)     | [Link](simulation/sp_fedavg_stackoverflow_lr_lr_example/README.md) |
| sp_fednova_mnist_lr_example                | Parrot                | FedNova             | MNIST             | Logistic Regression             | single process        | [Link](simulation/sp_fednova_mnist_lr_example)               | [Link](simulation/sp_fednova_mnist_lr_example/README.md)     |
| sp_fedopt_mnist_lr_example                 | Parrot                | FedOpt              | MNIST             | Logistic Regression             | single process        | [Link](simulation/sp_fedopt_mnist_lr_example)                | [Link](simulation/sp_fedopt_mnist_lr_example/README.md)      |
| sp_fedsgd_cifar10_resnet20_example         | Parrot                | FedSGD              | Cifar10           | Resnet20                        | single process        | [Link](simulation/sp_fedsgd_cifar10_resnet20_example)        | [Link](simulation/sp_fedsgd_cifar10_resnet20_example/README.md) |
| sp_hierarchicalfl_mnist_lr_example         | Parrot                | HierarchicalFL      | MNIST             | Logistic Regression             | single process        | [Link](simulation/sp_hierarchicalfl_mnist_lr_example)        | [Link](simulation/sp_hierarchicalfl_mnist_lr_example/README.md) |
| sp_turboaggregate_mnist_lr_example         | Parrot                | TurboAggregate      | MNIST             | Logistic Regression             | single process        | [Link](simulation/sp_turboaggregate_mnist_lr_example)        | [Link](simulation/sp_turboaggregate_mnist_lr_example/README.md) |
| sp_vertical_mnist_lr_example               | Parrot                | VerticalFL          | MNIST             | Logistic Regression             | single process        | [Link](simulation/sp_vertical_mnist_lr_example)              | [Link](simulation/sp_vertical_mnist_lr_example/README.md)    |
| mpi_base_framework_example                 | Octopus               | BaseFramework       | MNIST             | Logistic Regression             | MPI                   | [Link](simulation/mpi_base_framework_example)                | [Link](simulation/mpi_base_framework_example/README.md)      |
| mpi_decentralized_fl_example               | Octopus               | DecentralizedFL     | MNIST             | Logistic Regression             | MPI                   | [Link](simulation/mpi_decentralized_fl_example)              | [Link](simulation/mpi_decentralized_fl_example/README.md)    |
| mpi_fedavg_datasets_and_models_example     | Octopus               | FedAvg              | Cifar10           | MobileNet                       | MPI                   | [Link](simulation/mpi_fedavg_datasets_and_models_example)    | [Link](simulation/mpi_fedavg_datasets_and_models_example/README.md) |
|                                            | Octopus               | FedAvg              | Cifar100          | MobileNetV3                     | MPI                   | S/A                                                          | S/A                                                          |
|                                            | Octopus               | FedAvg              | Cinic             | Resnet56                        | MPI                   | S/A                                                          | S/A                                                          |
|                                            | Octopus               | FedAvg              | Fedcifar100       | Resnet18                        | MPI                   | S/A                                                          | S/A                                                          |
|                                            | Octopus               | FedAvg              | Fedemnist         | CNN                             | MPI                   | S/A                                                          | S/A                                                          |
|                                            | Octopus               | FedAvg              | Fedshakespeare    | RNN                             | MPI                   | S/A                                                          | S/A                                                          |
|                                            | Octopus               | FedAvg              | Lending_club_loan | Logistic Regression             | MPI                   | S/A                                                          | S/A                                                          |
|                                            | Octopus               | FedAvg              | MNIST             | CNN                             | MPI                   | S/A                                                          | S/A                                                          |
|                                            | Octopus               | FedAvg              | MNIST             | Logistic Regression             | MPI                   | S/A                                                          | S/A                                                          |
|                                            | Octopus               | FedAvg              | Shakespeare       | RNN                             | MPI                   | S/A                                                          | S/A                                                          |
|                                            | Octopus               | FedAvg              | Stackoverflow_lr  | Logistic Regression             | MPI                   | S/A                                                          | S/A                                                          |
|                                            | Octopus               | FedAvg              | Stackoverflow_nwp | RNN                             | MPI                   | S/A                                                          | S/A                                                          |
| mpi_fedavg_robust_example                  | Octopus               | Robust FedAvg       | Cifar10           | Resnet56                        | MPI                   | [Link](simulation/mpi_fedavg_robust_example)                 | [Link](simulation/mpi_fedavg_robust_example/README.md)       |
| mpi_fedopt_datasets_and_models_example     | Octopus               | FedOpt              | Cifar10           | MobileNet                       | MPI                   | [Link](simulation/mpi_fedopt_datasets_and_models_example)    | [Link](simulation/mpi_fedopt_datasets_and_models_example/README.md) |
|                                            | Octopus               | FedOpt              | Cifar100          | MobileNetV3                     | MPI                   | S/A                                                          | S/A                                                          |
|                                            | Octopus               | FedOpt              | Cinic             | Resnet56                        | MPI                   | S/A                                                          | S/A                                                          |
|                                            | Octopus               | FedOpt              | Fedcifar100       | Resnet18                        | MPI                   | S/A                                                          | S/A                                                          |
|                                            | Octopus               | FedOpt              | Fedemnist         | CNN                             | MPI                   | S/A                                                          | S/A                                                          |
|                                            | Octopus               | FedOpt              | Fedshakespeare    | RNN                             | MPI                   | S/A                                                          | S/A                                                          |
|                                            | Octopus               | FedOpt              | Lending_club_loan | Logistic Regression             | MPI                   | S/A                                                          | S/A                                                          |
|                                            | Octopus               | FedOpt              | MNIST             | CNN                             | MPI                   | S/A                                                          | S/A                                                          |
|                                            | Octopus               | FedOpt              | MNIST             | Logistic Regression             | MPI                   | S/A                                                          | S/A                                                          |
|                                            | Octopus               | FedOpt              | Shakespeare       | RNN                             | MPI                   | S/A                                                          | S/A                                                          |
|                                            | Octopus               | FedOpt              | Stackoverflow_lr  | Logistic Regression             | MPI                   | S/A                                                          | S/A                                                          |
|                                            | Octopus               | FedOpt              | Stackoverflow_nwp | RNN                             | MPI                   | S/A                                                          | S/A                                                          |
| mpi_fedprox_datasets_and_models_example    | Octopus               | FedProx             | Cifar10           | MobileNet                       | MPI                   | [Link](simulation/mpi_fedprox_datasets_and_models_example)   | [Link](simulation/mpi_fedprox_datasets_and_models_example/README.md) |
|                                            | Octopus               | FedProx             | Cifar100          | MobileNetV3                     | MPI                   | S/A                                                          | S/A                                                          |
|                                            | Octopus               | FedProx             | Cinic             | Resnet56                        | MPI                   | S/A                                                          | S/A                                                          |
|                                            | Octopus               | FedProx             | Fedcifar100       | Resnet18                        | MPI                   | S/A                                                          | S/A                                                          |
|                                            | Octopus               | FedProx             | Fedemnist         | CNN                             | MPI                   | S/A                                                          | S/A                                                          |
|                                            | Octopus               | FedProx             | Fedshakespeare    | RNN                             | MPI                   | S/A                                                          | S/A                                                          |
|                                            | Octopus               | FedProx             | Lending_club_loan | Logistic Regression             | MPI                   | S/A                                                          | S/A                                                          |
|                                            | Octopus               | FedProx             | MNIST             | CNN                             | MPI                   | S/A                                                          | S/A                                                          |
|                                            | Octopus               | FedProx             | MNIST             | Logistic Regression             | MPI                   | S/A                                                          | S/A                                                          |
|                                            | Octopus               | FedProx             | Shakespeare       | RNN                             | MPI                   | S/A                                                          | S/A                                                          |
|                                            | Octopus               | FedProx             | Stackoverflow_lr  | Logistic Regression             | MPI                   | S/A                                                          | S/A                                                          |
|                                            | Octopus               | FedProx             | Stackoverflow_nwp | RNN                             | MPI                   | S/A                                                          | S/A                                                          |
| mpi_torch_fedavg_mnist_lr_example          | Octopus               | FedAvg              | MNIST             | Logistic Regression             | MPI                   | [Link](simulation/mpi_torch_fedavg_mnist_lr_example)         | [Link](simulation/mpi_torch_fedavg_mnist_lr_example/README.md) |
| mpi_torch_fedgan_mnist_gan_example         | Octopus               | FedGAN              | MNIST             | Generating adversarial networks | MPI                   | [Link](simulation/mpi_torch_fedgan_mnist_gan_example)        | [Link](simulation/mpi_torch_fedgan_mnist_gan_example/README.md) |
| mpi_torch_fedgkt_mnist_lr_example          | Octopus               | FedGKT              | MNIST             | Logistic Regression             | MPI                   | [Link](simulation/mpi_torch_fedgkt_mnist_lr_example)         | [Link](simulation/mpi_torch_fedgkt_mnist_lr_example/README.md) |
| mpi_torch_fednas_cifar10_dart_example      | Octopus               | FedNAS              | Cifar10           | DART                            | MPI                   | [Link](simulation/mpi_torch_fednas_cifar10_dart_example)     | [Link](simulation/mpi_torch_fednas_cifar10_dart_example/README.md) |
| mpi_torch_fedopt_mnist_lr_example          | Octopus               | FedOpt              | MNIST             | Logistic Regression             | MPI                   | [Link](simulation/mpi_torch_fedopt_mnist_lr_example)         | [Link](simulation/mpi_torch_fedopt_mnist_lr_example/README.md) |
| mpi_torch_fedprox_mnist_lr_example         | Octopus               | FedProx             | MNIST             | Logistic Regression             | MPI                   | [Link](simulation/mpi_torch_fedprox_mnist_lr_example)        | [Link](simulation/mpi_torch_fedprox_mnist_lr_example/README.md) |
| mpi_torch_splitnn_cifar10_resnet56_example | Octopus               | SplitNN             | Cifar10           | ResNet56                        | MPI                   | [Link](simulation/mpi_torch_splitnn_cifar10_resnet56_example) | [Link](simulation/mpi_torch_splitnn_cifar10_resnet56_example/README.md) |
| nccl_fedavg_example                        | Octopus               | FedAvg              | Cifar10           | ResNet56                        | NCCL                  | [Link](simulation/nccl_fedavg_example)                       | [Link](simulation/nccl_fedavg_example/README.md)             |


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
