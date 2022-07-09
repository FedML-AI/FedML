# FedCV for Image Classification

## Model

Please customize the model by modifying the `model_args` section in `fedml_config.yaml` file.

Supported models:

- CNN
- Densenet
- Efficientnet
- MobileNetv2

## Dataset

Please customize the dataset by modifying the `data_args` section in `fedml_config.yaml` file.

Supported datasets:

- CIFAR10
- CIFAR100
- CINIC10
- FedCIFAR10
- FederatedEMNIST
- ImageNet
- Landmark
- MNIST

## How to use

### MPI Simulation

Install dependencies:

```bash
pip install fedml --upgrade
pip install mpi4py
```

For Linux

```bash
bash run_image_classification.sh 4
```

For Windows

```bash
mpiexec -np 5 python torch_mpi_image_classification.py --cf config\fedml_config.yaml
```

### Run on MLOps

1. Build package

```bash
pip install fedml --upgrade
bash build_mlops_package.sh
```

2. Create an application and upload package on MLOps

3. Login to MLOps

```bash
fedml login [$account_id]
```

4. Create a group and project
5. Create a run and select your device, then you can see the training logs and results
