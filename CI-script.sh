# code checking
pyflakes .

# activate the fedml environment
conda activate fedml

# test standalone
cd ./fedml_experiments/standalone/fedavg

# MNIST
wandb off
sh run_fedavg_standalone_pytorch.sh 2 10 10 mnist ./../../../data/mnist lr hetero 2 2 0.03

