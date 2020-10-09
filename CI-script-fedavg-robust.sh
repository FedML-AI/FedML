#!/bin/bash

set -ex

# code checking
pyflakes .

# activate the fedml environment
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate fedml

wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408
wandb off

# 1. CIFAR10 FedAvg-Robust
cd ./fedml_experiments/distributed/fedavg_robust
sh run_fedavg_robust_distributed_pytorch.sh 10 10 1 8 resnet56 homo 2 1 64 0.001 cifar10 "./../../../data/cifar10" weak-dp 5.0 0.025 &
cd ./../../../