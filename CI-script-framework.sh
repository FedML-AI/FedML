#!/bin/bash

set -ex

# code checking
pyflakes .

# activate the fedml environment
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate fedml

wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408
wandb off

# 1. distributed base framework
cd ./fedml_experiments/distributed/base
sh run_base_distributed_pytorch.sh &
cd ./../../../

# 2. decentralized base framework
cd ./fedml_experiments/distributed/decentralized_demo
sh run_decentralized_demo_distributed_pytorch.sh &
cd ./../../../
