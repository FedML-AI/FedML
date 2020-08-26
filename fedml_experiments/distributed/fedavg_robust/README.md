## Installation
http://doc.fedml.ai/#/installation-distributed-computing

## Run Experiments

### ResNet56 Federated Training

#### CIFAR10
train on IID dataset 
```
sh run_fedavg_robust_distributed_pytorch.sh 1 8 resnet56 homo 100 20 64 0.001 cifar10 "./../../../data/cifar10" weak-dp 5.0 0.025

##run on background
nohup sh run_fedavg_robust_distributed_pytorch.sh 1 8 resnet56 homo 100 20 64 0.001 cifar10 "./../../../data/cifar10" weak-dp 5.0 0.025 > ./fedavg-resnet-homo-cifar10.txt 2>&1 &
```

train on non-IID dataset
```
sh run_fedavg_robust_distributed_pytorch.sh 1 8 resnet56 hetero 100 20 64 0.001 cifar10 "./../../../data/cifar10" weak-dp 5.0 0.025

##run on background
nohup sh run_fedavg_robust_distributed_pytorch.sh 1 8 resnet56 hetero 100 20 64 0.001 cifar10 "./../../../data/cifar10" weak-dp 5.0 0.025 > ./fedavg-resnet-hetero-cifar10.txt 2>&1 &
```


#### CIFAR100
train on IID dataset 
```
sh run_fedavg_robust_distributed_pytorch.sh 1 8 resnet56 homo 100 20 64 0.001 cifar100 "./../../../data/cifar100" weak-dp 5.0 0.025

##run on background
nohup sh run_fedavg_robust_distributed_pytorch.sh 1 8 resnet56 homo 100 20 64 0.001 cifar100 "./../../../data/cifar100" weak-dp 5.0 0.025 > ./fedavg-resnet-homo-cifar100.txt 2>&1 &
```

train on non-IID dataset
```
sh run_fedavg_robust_distributed_pytorch.sh 1 8 resnet56 hetero 100 20 64 0.001 cifar100 "./../../../data/cifar100" weak-dp 5.0 0.025

##run on background
nohup sh run_fedavg_robust_distributed_pytorch.sh 1 8 resnet56 hetero 100 20 64 0.001 cifar100 "./../../../data/cifar100" weak-dp 5.0 0.025 > ./fedavg-resnet-hetero-cifar100.txt 2>&1 &
```


#### CINIC10
train on IID dataset 
```
sh run_fedavg_robust_distributed_pytorch.sh 1 8 resnet56 homo 100 20 64 0.001 cinic10 "./../../../data/cinic10" weak-dp 5.0 0.025

##run on background
nohup sh run_fedavg_robust_distributed_pytorch.sh 1 8 resnet56 homo 100 20 64 0.001 cinic10 "./../../../data/cinic10" weak-dp 5.0 0.025 > ./fedavg-resnet-homo-cinic10.txt 2>&1 &
```

train on non-IID dataset
```
sh run_fedavg_robust_distributed_pytorch.sh 1 8 resnet56 hetero 100 20 64 0.001 cinic10 "./../../../data/cinic10" weak-dp 5.0 0.025

##run on background
nohup sh run_fedavg_robust_distributed_pytorch.sh 1 8 resnet56 hetero 100 20 64 0.001 cinic10 "./../../../data/cinic10" weak-dp 5.0 0.025 > ./fedavg-resnet-hetero-cinic10.txt 2>&1 &
```


### MobileNet Federated Training

#### CIFAR10
train on IID dataset 
```
sh run_fedavg_robust_distributed_pytorch.sh 1 8 mobilenet homo 100 20 64 0.001 cifar10 "./../../../data/cifar10" weak-dp 5.0 0.025

##run on background
nohup sh run_fedavg_robust_distributed_pytorch.sh 1 8 mobilenet homo 100 20 64 0.001 cifar10 "./../../../data/cifar10" weak-dp 5.0 0.025 > ./fedavg-mobilenet-homo-cifar10.txt 2>&1 &
```

train on non-IID dataset
```
sh run_fedavg_robust_distributed_pytorch.sh 1 8 mobilenet hetero 100 20 64 0.001 cifar10 "./../../../data/cifar10" weak-dp 5.0 0.025

##run on background
nohup sh run_fedavg_robust_distributed_pytorch.sh 1 8 mobilenet hetero 100 20 64 0.001 cifar10 "./../../../data/cifar10" weak-dp 5.0 0.025 > ./fedavg-mobilenet-hetero-cifar10.txt 2>&1 &
```


#### CIFAR100
train on IID dataset 
```
sh run_fedavg_robust_distributed_pytorch.sh 1 8 mobilenet homo 100 20 64 0.001 cifar100 "./../../../data/cifar100" weak-dp 5.0 0.025

##run on background
nohup sh run_fedavg_robust_distributed_pytorch.sh 1 8 mobilenet homo 100 20 64 0.001 cifar100 "./../../../data/cifar100" weak-dp 5.0 0.025 > ./fedavg-mobilenet-homo-cifar100.txt 2>&1 &
```

train on non-IID dataset
```
sh run_fedavg_robust_distributed_pytorch.sh 1 8 mobilenet hetero 100 20 64 0.001 cifar100 "./../../../data/cifar100" weak-dp 5.0 0.025

##run on background
nohup sh run_fedavg_robust_distributed_pytorch.sh 1 8 mobilenet hetero 100 20 64 0.001 cifar100 "./../../../data/cifar100" weak-dp 5.0 0.025 > ./fedavg-mobilenet-hetero-cifar100.txt 2>&1 &
```


#### CINIC10
train on IID dataset 
```
sh run_fedavg_robust_distributed_pytorch.sh 1 8 mobilenet homo 100 20 64 0.001 cinic10 "./../../../data/cinic10" weak-dp 5.0 0.025

##run on background
nohup sh run_fedavg_robust_distributed_pytorch.sh 1 8 mobilenet homo 100 20 64 0.001 cinic10 "./../../../data/cinic10" weak-dp 5.0 0.025 > ./fedavg-mobilenet-homo-cinic10.txt 2>&1 &
```

train on non-IID dataset
```
sh run_fedavg_robust_distributed_pytorch.sh 1 8 mobilenet hetero 100 20 64 0.001 cinic10 "./../../../data/cinic10" weak-dp 5.0 0.025

##run on background
nohup sh run_fedavg_robust_distributed_pytorch.sh 1 8 mobilenet hetero 100 20 64 0.001 cinic10 "./../../../data/cinic10" weak-dp 5.0 0.025 > ./fedavg-mobilenet-hetero-cinic10.txt 2>&1 &
```
