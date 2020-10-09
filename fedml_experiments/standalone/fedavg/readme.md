## Experimental Tracking Platform (report real-time result to wandb.com, please change ID to your own)
wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408


## Experiment Scripts
Heterogeneous distribution (Non-IID) experiment:

Frond-end debugging:
``` 
# MNIST
sh run_fedavg_standalone_pytorch.sh 2 10 10 mnist ./../../../data/mnist lr hetero 200 20 0.03 sgd

# shakespeare
sh run_fedavg_standalone_pytorch.sh 2 10 10 shakespeare ./../../../data/shakespeare rnn hetero 100 10 0.8 sgd

# Federated EMNIST
sh run_fedavg_standalone_pytorch.sh 2 10 10 femnist ./../../../data/FederatedEMNIST cnn hetero 200 20 0.03 sgd

```

running on the background:
``` 
# MNIST
nohup sh run_fedavg_standalone_pytorch.sh 2 10 10 mnist ./../../../data/mnist lr hetero 200 20 0.03 sgd > ./fedavg_standalone.txt 2>&1 &

# shakespeare
nohup sh run_fedavg_standalone_pytorch.sh 2 10 10 shakespeare ./../../../data/shakespeare rnn hetero 100 10 0.8 sgd > ./fedavg_standalone.txt 2>&1 &

# Federated EMNIST
nohup sh run_fedavg_standalone_pytorch.sh 2 10 10 femnist ./../../../data/FederatedEMNIST cnn hetero 200 20 0.03 sgd > ./fedavg_standalone.txt 2>&1 &

```

For large DNNs (ResNet, Transformer, etc), please use the distributed computing (fedml_api/distributed). 


### Benchmark Results
We publish benchmark experimental results at wanb.com: \
https://app.wandb.ai/automl/fedml
