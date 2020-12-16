## Experimental Tracking Platform (report real-time result to wandb.com, please change ID to your own)
wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408

## Experiment results
| Dataset | Model | Accuracy |
| ------- | ------ | ------- |
| MNIST | cnn | 0.83 |
| Federated EMNIST | cnn | 0.75 |
| shakespeare (LEAF) | rnn | 0.53 |
| fed_shakespeare (Google) | rnn | 0.56 |
| fed_CIFAR100 | resnet | 0.35 |

## Experiment Scripts
Heterogeneous distribution (Non-IID) experiment:

Frond-end debugging:
``` 
## MNIST
sh run_fedavg_standalone_pytorch.sh 0 1000 10 10 mnist ./../../../data/mnist lr hetero 200 1 0.03 sgd 0
``` 
reference experimental result: https://wandb.ai/automl/fedml/runs/1ta35m0z

``` 
## shakespeare (LEAF)
sh run_fedavg_standalone_pytorch.sh 0 10 10 10 shakespeare ./../../../data/shakespeare rnn hetero 100 1 0.8 sgd 0
``` 
The experimental result refers to：https://wandb.ai/automl/fedml/runs/2381z9uv

``` 
# fed_shakespeare (Google)
sh run_fedavg_standalone_pytorch.sh 0 10 10 10 fed_shakespeare ./../../../data/fed_shakespeare/datasets rnn hetero 100 1 0.8 sgd 0
``` 
The experimental result refers to：https://wandb.ai/automl/fedml/runs/2nfj1ivc

``` 
## Federated EMNIST (with clip=1.0)
sh run_fedavg_standalone_pytorch.sh 0 10 10 20 femnist ./../../../data/FederatedEMNIST/datasets cnn hetero 300 1 0.03 sgd 0
``` 
The experimental result refers to：https://wandb.ai/automl/fedml/runs/wxn1f5gr

``` 
## Fed_CIFAR100
sh run_fedavg_standalone_pytorch.sh 5 10 10 10 fed_cifar100 ./../../../data/fed_cifar100/datasets resnet18_gn hetero 2000 1 0.1 sgd 0
```
The experimental result refers to：https://wandb.ai/automl/fedml/runs/34808l1v

``` 
# Stackoverflow_LR
sh run_fedavg_standalone_pytorch.sh 0 10 10 10 stackoverflow_lr ./../../../data/stackoverflow/datasets lr hetero 200 1 0.03 sgd 0
``` 

``` 
# Stackoverflow_NWP
sh run_fedavg_standalone_pytorch.sh 0 10 10 10 stackoverflow_nwp ./../../../data/stackoverflow/datasets rnn hetero 200 1 0.03 sgd 0
``` 

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
