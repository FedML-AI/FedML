#!/bin/bash

set -ex

# code checking
pyflakes .

# activate the fedml environment
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate fedml

wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408
wandb off

assert_eq() {
  local expected="$1"
  local actual="$2"
  local msg

  if [ "$expected" == "$actual" ]; then
    return 0
  else
    echo "$expected != $actual"
    return 1
  fi
}

round() {
  printf "%.${2}f" "${1}"
}

# 1. MNIST standalone FedAvg
cd ./fedml_experiments/standalone/fedavg
sh run_fedavg_standalone_pytorch.sh 0 2 2 4 mnist ./../../../data/mnist lr hetero 1 1 0.03 sgd 1
sh run_fedavg_standalone_pytorch.sh 0 2 2 4 shakespeare ./../../../data/shakespeare rnn hetero 1 1 0.8 sgd 1
sh run_fedavg_standalone_pytorch.sh 0 2 2 4 femnist ./../../../data/FederatedEMNIST/datasets cnn hetero 1 1 0.03 sgd 1
sh run_fedavg_standalone_pytorch.sh 0 2 2 4 fed_shakespeare ./../../../data/fed_shakespeare/datasets rnn hetero 1 1 0.8 sgd 1
sh run_fedavg_standalone_pytorch.sh 0 2 2 4 fed_cifar100 ./../../../data/fed_cifar100/datasets resnet18_gn hetero 1 1 0.03 adam 1
#sh run_fedavg_standalone_pytorch.sh 0 1 1 4 stackoverflow_lr ./../../../data/stackoverflow/datasets lr hetero 1 1 0.03 sgd 1
#sh run_fedavg_standalone_pytorch.sh 0 1 1 4 stackoverflow_nwp ./../../../data/stackoverflow/datasets cnn hetero 1 1 0.03 sgd 1

# assert that, for full batch and epochs=1, the accuracy of federated training(FedAvg) is equal to that of centralized training
sh run_fedavg_standalone_pytorch.sh 0 1 1 -1 mnist ./../../../data/mnist lr hetero 10 1 0.03 sgd 0
centralized_full_train_acc=$(cat wandb/latest-run/files/wandb-summary.json | python -c "import sys, json; print(json.load(sys.stdin)['Train/Acc'])")
sh run_fedavg_standalone_pytorch.sh 0 1000 1000 -1 mnist ./../../../data/mnist lr hetero 10 1 0.03 sgd 0
federated_full_train_acc=$(cat wandb/latest-run/files/wandb-summary.json | python -c "import sys, json; print(json.load(sys.stdin)['Train/Acc'])")
assert_eq $(round $centralized_full_train_acc 3) $(round $federated_full_train_acc 3)
cd ./../../../

# assert that, for full batch and epochs=1 and when the product of global and group comm. round is fixed,
# the accuracy of hierarchical federated learning is equal to that of centralized training, regardless of the number of groups
cd ./fedml_experiments/standalone/hierarchical_fl
sh run_standalone_pytorch.sh 0 1000 1000 -1 mnist ./../../../data/mnist lr hetero 0.03 sgd random 2 5 2 1
hierarchical_fl_full_train_acc=$(cat wandb/latest-run/files/wandb-summary.json | python -c "import sys, json; print(json.load(sys.stdin)['Train/Acc'])")
assert_eq $(round $centralized_full_train_acc 3) $(round $hierarchical_fl_full_train_acc 3)
sh run_standalone_pytorch.sh 0 1000 1000 -1 mnist ./../../../data/mnist lr hetero 0.03 sgd random 2 2 5 1
hierarchical_fl_full_train_acc=$(cat wandb/latest-run/files/wandb-summary.json | python -c "import sys, json; print(json.load(sys.stdin)['Train/Acc'])")
assert_eq $(round $centralized_full_train_acc 3) $(round $hierarchical_fl_full_train_acc 3)
cd ./../../../


# 2. MNIST distributed FedAvg
#cd ./fedml_experiments/distributed/fedavg
#sh run_fedavg_distributed_pytorch.sh 2 2 1 4 lr hetero 1 1 2 0.03 mnist "./../../../data/mnist" 1
#sh run_fedavg_distributed_pytorch.sh 2 2 1 4 cnn hetero 1 1 10 0.8 femnist "./../../../data/FederatedEMNIST/datasets" 1
#sh run_fedavg_distributed_pytorch.sh 2 2 1 4 resnet18_gn hetero 1 1 10 0.8 fed_cifar100 "./../../../data/fed_cifar100/datasets" 1
#sh run_fedavg_distributed_pytorch.sh 2 2 1 4 rnn hetero 1 1 10 0.8 shakespeare "./../../../data/shakespeare" 1
#sh run_fedavg_distributed_pytorch.sh 2 2 1 4 rnn hetero 1 1 10 0.8 fed_shakespeare "./../../../data/fed_shakespeare/datasets" 1

#sh run_fedavg_distributed_pytorch.sh 2 2 1 4 resnet56 homo 1 1 64 0.001 cifar10 "./../../../data/cifar10" 1
#sh run_fedavg_distributed_pytorch.sh 2 2 1 4 resnet56 hetero 1 1 64 0.001 cifar10 "./../../../data/cifar10" 1
#sh run_fedavg_distributed_pytorch.sh 2 2 1 4 resnet56 homo 1 1 64 0.001 cifar100 "./../../../data/cifar100" 1
#sh run_fedavg_distributed_pytorch.sh 2 2 1 4 resnet56 hetero 1 1 64 0.001 cifar100 "./../../../data/cifar100" 1
#sh run_fedavg_distributed_pytorch.sh 2 2 1 4 resnet56 homo 1 1 64 0.001 cinic10 "./../../../data/cinic10" 1
#sh run_fedavg_distributed_pytorch.sh 2 2 1 4 resnet56 hetero 1 1 64 0.001 cinic10 "./../../../data/cinic10" 1
#cd ./../../../

# 3. MNIST mobile FedAvg
#cd ./fedml_mobile/server/executor/
#python3 app.py &
#bg_pid_server=$!
#echo "pid="$bg_pid_server
#
#sleep 30
#python3 ./mobile_client_simulator.py --client_uuid '0' &
#bg_pid_client0=$!
#echo $bg_pid_client0
#
#python3 ./mobile_client_simulator.py --client_uuid '1' &
#bg_pid_client1=$!
#echo $bg_pid_client1
#
#sleep 80
#kill $bg_pid_server
#kill $bg_pid_client0
#kill $bg_pid_client1

#cd ./../../../
