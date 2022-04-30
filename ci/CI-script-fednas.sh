#!/bin/bash

set -ex

# code checking
pyflakes .

# activate the fedml environment
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate fedml

wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408
wandb off

# 1. search
cd ./fedml_experiments/distributed/fednas/
sh run_fednas_search.sh 1 4 darts homo 2 1 2 &
cd ./../../../

# 2. train
cd ./fedml_experiments/distributed/fednas/
sh run_fednas_train.sh 1 4 darts homo 2 1 2 &
cd ./../../../