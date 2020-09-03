#!/bin/bash

set -ex

# code checking
pyflakes .

# activate the fedml environment
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate fedml

wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408
wandb off


# 1. MNIST standalone FedAvg
cd ./fedml_experiments/standalone/fedavg
sh run_fedavg_standalone_pytorch.sh 2 10 10 mnist ./../../../data/mnist lr hetero 2 2 0.03
cd ./../../../



# 2. MNIST distributed FedAvg
cd ./fedml_experiments/distributed/fedavg
sh run_fedavg_distributed_pytorch.sh 10 10 1 4 lr hetero 2 2 10 0.03 mnist "./../../../data/mnist" & || exit 1

sleep 60
killall mpirun
cd ./../../../

# 3. MNIST mobile FedAvg
cd ./fedml_mobile/server/executor/
python3 app.py &
bg_pid_server=$!
echo "pid="$bg_pid_server

sleep 30
python3 ./mobile_client_simulator.py --client_uuid '0' &
bg_pid_client0=$!
echo $bg_pid_client0

python3 ./mobile_client_simulator.py --client_uuid '1' &
bg_pid_client1=$!
echo $bg_pid_client1

sleep 80
kill $bg_pid_server
kill $bg_pid_client0
kill $bg_pid_client1

cd ./../../../
