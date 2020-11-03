## Installation
http://doc.fedml.ai/#/installation-distributed-computing

## Supported Attack and Defense Methods:
### Attacks:
1. The "Edge-case" Black-box Attack (NeurIPS 2020) [Link](https://arxiv.org/abs/2007.05084)
2. (TODO) The "Edge-case" PGD Attack with Model Replacement (NeurIPS 2020) [Link](https://arxiv.org/abs/2007.05084)
3. (TODO) The "Edge-case" PGD Attack without Model Replacement (NeurIPS 2020) [Link](https://arxiv.org/abs/2007.05084)

### Defenses:
1. The Norm Difference Clipping Defense [Link](https://arxiv.org/abs/1911.07963)
2. The Weakly Differentialy Private Defense (Weak-DP) [Link](https://arxiv.org/abs/1911.07963)
3. (TODO) The RFA Defense [Link](https://arxiv.org/abs/1912.13445)
4. (TODO) The Krum/Multi-Krum Defenses (NeurIPS 2017) [Link](https://papers.nips.cc/paper/6617-machine-learning-with-adversaries-byzantine-tolerant-gradient-descent)

## Run Experiments

### ResNet56 Federated Training

#### CIFAR10 + Southwest Edge Case Backdoor (Black-box) under Weak-DP Defense
train on IID dataset 
```
sh run_fedavg_robust_distributed_pytorch.sh 1 4 resnet56 homo 100 20 64 0.001 cifar10 "./../../../data/cifar10" weak-dp 5.0 0.025 southwest 10

##run on background
nohup sh run_fedavg_robust_distributed_pytorch.sh 1 8 resnet56 homo 100 20 64 0.001 cifar10 "./../../../data/cifar10" weak-dp 5.0 0.025 southwest 10 > ./fedavg-resnet-homo-cifar10.txt 2>&1 &
```

train on non-IID dataset
```
sh run_fedavg_robust_distributed_pytorch.sh 1 8 resnet56 hetero 100 20 64 0.001 cifar10 "./../../../data/cifar10" weak-dp 5.0 0.025 southwest 10

##run on background
nohup sh run_fedavg_robust_distributed_pytorch.sh 1 8 resnet56 hetero 100 20 64 0.001 cifar10 "./../../../data/cifar10" weak-dp 5.0 0.025 southwest 10 > ./fedavg-resnet-hetero-cifar10.txt 2>&1 &
```