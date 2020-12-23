## Experimental Tracking Platform (report real-time result to wandb.com, please change ID to your own)
wandb login {YOUR_WANDB_API_KEY}

## Experiment results
| Dataset | Model | Accuracy (Exp/Ref)|
| ------- | ------ | ------- |
| MNIST | cnn | 81.9 / |
| Federated EMNIST | cnn | 80.2 / 84.9 |
| fed_CIFAR100 | resnet | 34.0 / 44.7|
| shakespeare (LEAF) | rnn | 53.1 /  |
| fed_shakespeare (Google) | rnn | 57.1 / 56.9 |
| stackoverflow_nwp | rnn | 18.3 / 19.5 |
(Exp results are the test accuracy of the last communication rounds, while the reference results are the validation results from referenced paper.)

## Experiment Scripts
Heterogeneous distribution (Non-IID) experiment:

Frond-end debugging:
``` 
## MNIST
sh run_fedavg_standalone_pytorch.sh 0 1000 10 10 mnist ./../../../data/mnist lr hetero 200 1 0.03 sgd 0
``` 
reference experimental result: https://app.wandb.ai/automl/fedml/runs/ybv29kak

``` 
## shakespeare (LEAF)
sh run_fedavg_standalone_pytorch.sh 0 10 10 10 shakespeare ./../../../data/shakespeare rnn hetero 100 1 0.8 sgd 0
``` 
The experimental result refers to：https://app.wandb.ai/automl/fedml/runs/2al5q5mi

``` 
# fed_shakespeare (Google)
sh run_fedavg_standalone_pytorch.sh 0 10 10 10 fed_shakespeare ./../../../data/fed_shakespeare/datasets rnn hetero 1000 1 0.8 sgd 0
``` 
The experimental result refers to：https://wandb.ai/automl/fedml/runs/4btyrt0u

``` 
## Federated EMNIST 
sh run_fedavg_standalone_pytorch.sh 0 10 10 20 femnist ./../../../data/FederatedEMNIST/datasets cnn hetero 1000 1 0.03 sgd 0
``` 
The experimental result refers to：https://wandb.ai/automl/fedml/runs/3lv4gmpz

``` 
## Fed_CIFAR100
sh run_fedavg_standalone_pytorch.sh 0 10 10 10 fed_cifar100 ./../../../data/fed_cifar100/datasets resnet18_gn hetero 4000 1 0.1 sgd 0
```
The experimental result refers to：https://wandb.ai/automl/fedml/runs/1canbwed

``` 
# Stackoverflow_LR
sh run_fedavg_standalone_pytorch.sh 0 10 10 10 stackoverflow_lr ./../../../data/stackoverflow/datasets lr hetero 2000 1 0.03 sgd 0
# https://wandb.ai/automl/fedml/runs/3aponqml
``` 

``` 
# Stackoverflow_NWP
sh run_fedavg_standalone_pytorch.sh 0 10 10 10 stackoverflow_nwp ./../../../data/stackoverflow/datasets rnn hetero 2000 1 0.03 sgd 0
``` 
The experimental result refers to: https://wandb.ai/automl/fedml/runs/7pf2c9r2

``` 
# CIFAR10
sh run_fedavg_standalone_pytorch.sh 0 10 10 10 cifar10 ./../../../data/cifar10 resnet56 hetero 200 1 0.03 sgd 0
```

Please make sure to run on the background when you start training after debugging. An example to run on the background:
``` 
# MNIST
nohup sh run_fedavg_standalone_pytorch.sh 2 10 10 10 mnist ./../../../data/mnist lr hetero 200 20 0.03 sgd 0 > ./fedavg_standalone.txt 2>&1 &
```

For large DNNs (ResNet, Transformer, etc), please use the distributed computing (fedml_api/distributed). 


### Benchmark Results
We publish benchmark experimental results at wanb.com: \
https://app.wandb.ai/automl/fedml
