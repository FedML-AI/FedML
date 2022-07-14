# Examples

- This is the outline of all examples. For more detailed instructions, please refer to [https://doc.fedml.ai](https://doc.fedml.ai)
- In [FedML/python/app](./../app) folder, we also provide applications in real-world settings.



## Cross-silo Federated Learning for cross-organization/account training

|                                 | platform/scenario    | federated optimizer | dataset | model               | communication backend | source code                                                  | example doc                                                  |
| ------------------------------- | -------------------- | ------------------- | ------- | ------------------- | --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| mqtt_s3_fedavg_mnist_lr_example | Octopus (cross-silo) | FedAvg              | MNIST   | Logistic Regression | MQTT_S3               | [Link](cross_silo/mqtt_s3_fedavg_mnist_lr_example)           | [Link](cross_silo/mqtt_s3_fedavg_mnist_lr_example/README.md) |
| grpc_fedavg_mnist_lr_example    | Octopus (cross-silo) | FedAvg              | MNIST   | Logistic Regression | GRPC                  | [Link](cross_silo/grpc_fedavg_mnist_lr_example)              | [Link](cross_silo/grpc_fedavg_mnist_lr_example/README.md)    |
| mpi_fedavg_mnist_lr_example     | Octopus (cross-silo) | FedAvg              | MNIST   | Logistic Regression | MPI                   | [Link](cross_silo/mpi_fedavg_mnist_lr_example)               | [Link](cross_silo/mpi_fedavg_mnist_lr_example/README.md)     |
| trpc_fedavg_mnist_lr_example    | Octopus (cross-silo) | FedAvg              | MNIST   | Logistic Regression | TRPC                  | [Link](cross_silo/trpc_fedavg_mnist_lr_example)              | [Link](cross_silo/trpc_fedavg_mnist_lr_example/README.md)    |
| mqtt_s3_fedavg_mnist_lr_example | Octopus (cross-silo) | FedAvg              | MNIST   | Logistic Regression | CUDA RPC              | [Link](cross_silo/cuda_rpc_fedavg_mnist_lr_example)          | [Link](cross_silo/cuda_rpc_fedavg_mnist_lr_example/README.md) |
| grpc_fedavg_mnist_lr_example    | Octopus (cross-silo) | FedAvg              | MNIST   | Logistic Regression | MQTT_S3               | [Link](cross_silo/mqtt_s3_fedavg_hierarchical_mnist_lr_example) | [Link](cross_silo/mqtt_s3_fedavg_hierarchical_mnist_lr_example/README.md) |


## Cross-device Federated Learning for Smartphones

|                                 | platform/scenario     | federated optimizer | dataset | model               | communication backend | source code                                          | example doc                                                  |
| ------------------------------- | --------------------- | ------------------- | ------- | ------------------- | --------------------- | ---------------------------------------------------- | ------------------------------------------------------------ |
| mqtt_s3_fedavg_mnist_lr_example | Beehive(Cross-device) | FedAvg              | MNIST   | Logistic Regression | MQTT_S3               | [Link](cross_device/mqtt_s3_fedavg_mnist_lr_example) | [Link](cross_device/mqtt_s3_fedavg_mnist_lr_example/README.md) |


## Simulation: (1) simulate FL using a single process (2) MPI-based FL Simulator (3) NCCL-based FL Simulator (fastest)
|                                            | platform/scenario  | federated optimizer | dataset          | model                           | communication backend | source code                                                  | example doc                                                  |
| ------------------------------------------ | ------------------ | ------------------- | ---------------- | ------------------------------- | --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| sp_decentralized_mnist_lr_example          | Parrot (simulator) | DecentralizedFL     | MNIST            | Logistic Regression             | single process        | [Link](simulation/sp_decentralized_mnist_lr_example)         | [Link](simulation/sp_decentralized_mnist_lr_example/README.md) |
| sp_fedavg_cifar10_cnn_example              | Parrot (simulator) | FedAvg              | Cifar10          | CNN                             | single process        | [Link](simulation/sp_fedavg_cifar10_cnn_example)             | [Link](simulation/sp_fedavg_cifar10_cnn_example/README.md)   |
| sp_fedavg_cifar10_mobilenet_example        | Parrot (simulator) | FedAvg              | Cifar10          | MobileNet                       | single process        | [Link](simulation/sp_fedavg_cifar10_mobilenet_example)       | [Link](simulation/sp_fedavg_cifar10_mobilenet_example/README.md) |
| sp_fedavg_cifar10_resnet56_example         | Parrot (simulator) | FedAvg              | Cifar10          | Resnet56                        | single process        | [Link](simulation/sp_fedavg_cifar10_resnet56_example)        | [Link](simulation/sp_fedavg_cifar10_resnet56_example/README.md) |
| sp_fedavg_mnist_lr_example                 | Parrot (simulator) | FedAvg              | MNIST            | Logistic Regression             | single process        | [Link](simulation/sp_fedavg_mnist_lr_example)                | [Link](simulation/sp_fedavg_mnist_lr_example/README.md)      |
| sp_fedavg_stackoverflow_lr_lr_example      | Parrot (simulator) | FedAvg              | Stackoverflow_lr | Logistic Regression             | single process        | [Link](simulation/sp_fedavg_stackoverflow_lr_lr_example)     | [Link](simulation/sp_fedavg_stackoverflow_lr_lr_example/README.md) |
| sp_fednova_mnist_lr_example                | Parrot (simulator) | FedNova             | MNIST            | Logistic Regression             | single process        | [Link](simulation/sp_fednova_mnist_lr_example)               | [Link](simulation/sp_fednova_mnist_lr_example/README.md)     |
| sp_fedopt_mnist_lr_example                 | Parrot (simulator) | FedOpt              | MNIST            | Logistic Regression             | single process        | [Link](simulation/sp_fedopt_mnist_lr_example)                | [Link](simulation/sp_fedopt_mnist_lr_example/README.md)      |
| sp_fedsgd_cifar10_resnet20_example         | Parrot (simulator) | FedSGD              | Cifar10          | Resnet20                        | single process        | [Link](simulation/sp_fedsgd_cifar10_resnet20_example)        | [Link](simulation/sp_fedsgd_cifar10_resnet20_example/README.md) |
| sp_hierarchicalfl_mnist_lr_example         | Parrot (simulator) | HierarchicalFL      | MNIST            | Logistic Regression             | single process        | [Link](simulation/sp_hierarchicalfl_mnist_lr_example)        | [Link](simulation/sp_hierarchicalfl_mnist_lr_example/README.md) |
| sp_turboaggregate_mnist_lr_example         | Parrot (simulator) | TurboAggregate      | MNIST            | Logistic Regression             | single process        | [Link](simulation/sp_turboaggregate_mnist_lr_example)        | [Link](simulation/sp_turboaggregate_mnist_lr_example/README.md) |
| sp_vertical_mnist_lr_example               | Parrot (simulator) | VerticalFL          | MNIST            | Logistic Regression             | single process        | [Link](simulation/sp_vertical_mnist_lr_example)              | [Link](simulation/sp_vertical_mnist_lr_example/README.md)    |
| mpi_base_framework_example                 | Parrot (simulator) | BaseFramework       | MNIST            | Logistic Regression             | MPI                   | [Link](simulation/mpi_base_framework_example)                | [Link](simulation/mpi_base_framework_example/README.md)      |
| mpi_decentralized_fl_example               | Parrot (simulator) | DecentralizedFL     | MNIST            | Logistic Regression             | MPI                   | [Link](simulation/mpi_decentralized_fl_example)              | [Link](simulation/mpi_decentralized_fl_example/README.md)    |
| mpi_fedavg_datasets_and_models_example     | Parrot (simulator) | FedAvg              | Cifar10          | MobileNet                       | MPI                   | [Link](simulation/mpi_fedavg_datasets_and_models_example)    | [Link](simulation/mpi_fedavg_datasets_and_models_example/README.md) |
| mpi_fedavg_robust_example                  | Parrot (simulator) | Robust FedAvg       | Cifar10          | Resnet56                        | MPI                   | [Link](simulation/mpi_fedavg_robust_example)                 | [Link](simulation/mpi_fedavg_robust_example/README.md)       |
| mpi_fedopt_datasets_and_models_example     | Parrot (simulator) | FedOpt              | Cifar10          | MobileNet                       | MPI                   | [Link](simulation/mpi_fedopt_datasets_and_models_example)    | [Link](simulation/mpi_fedopt_datasets_and_models_example/README.md) |
| mpi_fedprox_datasets_and_models_example    | Parrot (simulator) | FedProx             | Cifar10          | MobileNet                       | MPI                   | [Link](simulation/mpi_fedprox_datasets_and_models_example)   | [Link](simulation/mpi_fedprox_datasets_and_models_example/README.md) |
| mpi_torch_fedavg_mnist_lr_example          | Parrot (simulator) | FedAvg              | MNIST            | Logistic Regression             | MPI                   | [Link](simulation/mpi_torch_fedavg_mnist_lr_example)         | [Link](simulation/mpi_torch_fedavg_mnist_lr_example/README.md) |
| mpi_torch_fedgan_mnist_gan_example         | Parrot (simulator) | FedGAN              | MNIST            | Generating adversarial networks | MPI                   | [Link](simulation/mpi_torch_fedgan_mnist_gan_example)        | [Link](simulation/mpi_torch_fedgan_mnist_gan_example/README.md) |
| mpi_torch_fedgkt_mnist_lr_example          | Parrot (simulator) | FedGKT              | MNIST            | Logistic Regression             | MPI                   | [Link](simulation/mpi_torch_fedgkt_mnist_lr_example)         | [Link](simulation/mpi_torch_fedgkt_mnist_lr_example/README.md) |
| mpi_torch_fednas_cifar10_dart_example      | Parrot (simulator) | FedNAS              | Cifar10          | DART                            | MPI                   | [Link](simulation/mpi_torch_fednas_cifar10_dart_example)     | [Link](simulation/mpi_torch_fednas_cifar10_dart_example/README.md) |
| mpi_torch_fedopt_mnist_lr_example          | Parrot (simulator) | FedOpt              | MNIST            | Logistic Regression             | MPI                   | [Link](simulation/mpi_torch_fedopt_mnist_lr_example)         | [Link](simulation/mpi_torch_fedopt_mnist_lr_example/README.md) |
| mpi_torch_fedprox_mnist_lr_example         | Parrot (simulator) | FedProx             | MNIST            | Logistic Regression             | MPI                   | [Link](simulation/mpi_torch_fedprox_mnist_lr_example)        | [Link](simulation/mpi_torch_fedprox_mnist_lr_example/README.md) |
| mpi_torch_splitnn_cifar10_resnet56_example | Parrot (simulator) | SplitNN             | Cifar10          | ResNet56                        | MPI                   | [Link](simulation/mpi_torch_splitnn_cifar10_resnet56_example) | [Link](simulation/mpi_torch_splitnn_cifar10_resnet56_example/README.md) |
| nccl_fedavg_example                        | Parrot (simulator) | FedAvg              | Cifar10          | ResNet56                        | NCCL                  | [Link](simulation/nccl_fedavg_example)                       | [Link](simulation/nccl_fedavg_example/README.md)             |

## Distributed Training: Accelerate Model Training with Lightweight Cheetah

Coming soon
