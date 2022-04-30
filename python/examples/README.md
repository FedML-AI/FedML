# Distributed Training: Accelerate Model Training with Lightweight Cheetah


# Simulation: (1) simulate FL using a single process (2) MPI-based FL Simulator (3) NCCL-based FL Simulator (fastest)

### (1) Single Process
- FedAvg on MNIST dataset with Logistic Regression model (tested): `python/examples/simulation/sp_fedavg_mnist_lr_example`

### (2) MPI-based Simulator
- FedAvg on MNIST dataset with Logistic Regression model (tested): `python/examples/simulation/mpi_torch_fedavg_mnist_lr_example`

# Cross-silo Federated Learning for cross-organization/account training

using communication backend MQTT_S3 (tested): `python/examples/cross_silo/mqtt_s3_fedavg_mnist_lr_example`

using communication backend gRPC (not tested yet): `python/examples/cross_silo/mqtt_s3_fedavg_mnist_lr_example`

using communication backend PyTorch RPC (not tested yet): `python/examples/cross_silo/mqtt_s3_fedavg_mnist_lr_example`


# Cross-device Federated Learning for Smartphones and IoTs

using communication backend MQTT_S3_MNN (tested): `python/examples/cross_device/mqtt_s3_fedavg_mnist_lr_example`