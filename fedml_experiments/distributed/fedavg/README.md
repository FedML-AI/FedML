## Installation
http://doc.fedml.ai/#/installation-distributed-computing

## Run Experiments

## MNIST experiments
```
sh run_fedavg_distributed_pytorch.sh 10 10 1 4 lr hetero 200 20 10 0.03 mnist "./../../../data/mnist" 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 1 4 lr hetero 200 20 10 0.03 mnist "./../../../data/mnist" 0 > ./fedavg-lr-mnist.txt 2>&1 &
```

# Federated EMNIST experiments
```
sh run_fedavg_distributed_pytorch.sh 10 10 1 4 rnn hetero 100 10 10 0.8 femnist "./../../../data/FederatedEMNIST" 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 1 4 rnn hetero 100 10 10 0.8 femnist "./../../../data/FederatedEMNIST" 0 > ./fedavg-cnn-femnist.txt 2>&1 &
```

# shakespeare experiments
```
sh run_fedavg_distributed_pytorch.sh 10 10 1 4 rnn hetero 100 10 10 0.8 shakespeare "./../../../data/shakespeare" 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 1 4 rnn hetero 100 10 10 0.8 shakespeare "./../../../data/shakespeare" 0 > ./fedavg-rnn-shakespeare.txt 2>&1 &
```

## ResNet56 Federated Training

#### CIFAR10
train on IID dataset 
```
sh run_fedavg_distributed_pytorch.sh 10 10 1 8 resnet56 homo 100 20 64 0.001 cifar10 "./../../../data/cifar10" 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 1 8 resnet56 homo 100 20 64 0.001 cifar10 "./../../../data/cifar10" 0 > ./fedavg-resnet-homo-cifar10.txt 2>&1 &
```

train on non-IID dataset
```
sh run_fedavg_distributed_pytorch.sh 10 10 1 8 resnet56 hetero 100 20 64 0.001 cifar10 "./../../../data/cifar10" 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 1 8 resnet56 hetero 100 20 64 0.001 cifar10 "./../../../data/cifar10" 0 > ./fedavg-resnet-hetero-cifar10.txt 2>&1 &
```


#### CIFAR100
train on IID dataset 
```
sh run_fedavg_distributed_pytorch.sh 10 10 1 8 resnet56 homo 100 20 64 0.001 cifar100 "./../../../data/cifar100" 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 1 8 resnet56 homo 100 20 64 0.001 cifar100 "./../../../data/cifar100" 0 > ./fedavg-resnet-homo-cifar100.txt 2>&1 &
```

train on non-IID dataset
```
sh run_fedavg_distributed_pytorch.sh 10 10 1 8 resnet56 hetero 100 20 64 0.001 cifar100 "./../../../data/cifar100" 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 1 8 resnet56 hetero 100 20 64 0.001 cifar100 "./../../../data/cifar100" 0 > ./fedavg-resnet-hetero-cifar100.txt 2>&1 &
```


#### CINIC10
train on IID dataset 
```
sh run_fedavg_distributed_pytorch.sh 10 10 1 8 resnet56 homo 100 20 64 0.001 cinic10 "./../../../data/cinic10" 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 1 8 resnet56 homo 100 20 64 0.001 cinic10 "./../../../data/cinic10" 0 > ./fedavg-resnet-homo-cinic10.txt 2>&1 &
```

train on non-IID dataset
```
sh run_fedavg_distributed_pytorch.sh 10 10 1 8 resnet56 hetero 100 20 64 0.001 cinic10 "./../../../data/cinic10" 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 1 8 resnet56 hetero 100 20 64 0.001 cinic10 "./../../../data/cinic10" 0 > ./fedavg-resnet-hetero-cinic10.txt 2>&1 &
```


## MobileNet Federated Training

#### CIFAR10
train on IID dataset 
```
sh run_fedavg_distributed_pytorch.sh 10 10 1 8 mobilenet homo 100 20 64 0.001 cifar10 "./../../../data/cifar10" 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 1 8 mobilenet homo 100 20 64 0.001 cifar10 "./../../../data/cifar10" 0 > ./fedavg-mobilenet-homo-cifar10.txt 2>&1 &
```

train on non-IID dataset
```
sh run_fedavg_distributed_pytorch.sh 10 10 1 8 mobilenet hetero 100 20 64 0.001 cifar10 "./../../../data/cifar10" 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 1 8 mobilenet hetero 100 20 64 0.001 cifar10 "./../../../data/cifar10" 0 > ./fedavg-mobilenet-hetero-cifar10.txt 2>&1 &
```


#### CIFAR100
train on IID dataset 
```
sh run_fedavg_distributed_pytorch.sh 10 10 1 8 mobilenet homo 100 20 64 0.001 cifar100 "./../../../data/cifar100" 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 1 8 mobilenet homo 100 20 64 0.001 cifar100 "./../../../data/cifar100" 0 > ./fedavg-mobilenet-homo-cifar100.txt 2>&1 &
```

train on non-IID dataset
```
sh run_fedavg_distributed_pytorch.sh 10 10 1 8 mobilenet hetero 100 20 64 0.001 cifar100 "./../../../data/cifar100" 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 1 8 mobilenet hetero 100 20 64 0.001 cifar100 "./../../../data/cifar100" 0 > ./fedavg-mobilenet-hetero-cifar100.txt 2>&1 &
```


#### CINIC10
train on IID dataset 
```
sh run_fedavg_distributed_pytorch.sh 10 1 8 mobilenet homo 100 20 64 0.001 cinic10 "./../../../data/cinic10" 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 1 8 mobilenet homo 100 20 64 0.001 cinic10 "./../../../data/cinic10" 0 > ./fedavg-mobilenet-homo-cinic10.txt 2>&1 &
```

train on non-IID dataset
```
sh run_fedavg_distributed_pytorch.sh 10 1 8 mobilenet hetero 100 20 64 0.001 cinic10 "./../../../data/cinic10" 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 1 8 mobilenet hetero 100 20 64 0.001 cinic10 "./../../../data/cinic10" 0 > ./fedavg-mobilenet-hetero-cinic10.txt 2>&1 &
```
