## Installation
http://doc.fedml.ai/#/installation-distributed-computing

## Run Experiments
Note please tune hyper-parameters accordingly. 
You can refer the hyper-parameter suggestions at `FedML/benchmark/README.md`), but this may not be the optimal.

## MNIST experiments
```
sh run_fedavg_distributed_pytorch.sh 1000 10 1 4 lr hetero 200 1 10 0.03 mnist "./../../../data/mnist" sgd 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 1000 10 1 4 lr hetero 200 1 10 0.03 mnist "./../../../data/mnist" sgd 0 > ./fedavg-lr-mnist.txt 2>&1 &
```
The reference experimental results using the above hyper-parameters:  
https://wandb.ai/automl/fedml/runs/2dntp1tv?workspace=user-chaoyanghe-com


# Federated EMNIST experiments
```
sh run_fedavg_distributed_pytorch.sh 10 10 1 4 cnn hetero 100 1 20 0.1 femnist "./../../../data/FederatedEMNIST" sgd 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 1 4 cnn hetero 100 10 20 0.1 femnist "./../../../data/FederatedEMNIST" sgd 0 > ./fedavg-cnn-femnist.txt 2>&1 &
```

# shakespeare experiments
```
sh run_fedavg_distributed_pytorch.sh 10 10 1 4 rnn hetero 100 10 10 0.8 shakespeare "./../../../data/shakespeare" sgd 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 1 4 rnn hetero 100 10 10 0.8 shakespeare "./../../../data/shakespeare" sgd 0 > ./fedavg-rnn-shakespeare.txt 2>&1 &
```

## ResNet56 Federated Training

#### CIFAR10
train on IID dataset 
```
sh run_fedavg_distributed_pytorch.sh 10 10 1 8 resnet56 homo 100 20 64 0.001 cifar10 "./../../../data/cifar10" adam 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 1 8 resnet56 homo 100 20 64 0.001 cifar10 "./../../../data/cifar10" adam 0 > ./fedavg-resnet-homo-cifar10.txt 2>&1 &
```

train on non-IID dataset
```
sh run_fedavg_distributed_pytorch.sh 10 10 1 8 resnet56 hetero 100 20 64 0.001 cifar10 "./../../../data/cifar10" adam 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 1 8 resnet56 hetero 100 20 64 0.001 cifar10 "./../../../data/cifar10" adam 0 > ./fedavg-resnet-hetero-cifar10.txt 2>&1 &
```


#### CIFAR100
train on IID dataset 
```
sh run_fedavg_distributed_pytorch.sh 10 10 1 8 resnet56 homo 100 20 64 0.001 cifar100 "./../../../data/cifar100" adam 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 1 8 resnet56 homo 100 20 64 0.001 cifar100 "./../../../data/cifar100" adam 0 > ./fedavg-resnet-homo-cifar100.txt 2>&1 &
```

train on non-IID dataset
```
sh run_fedavg_distributed_pytorch.sh 10 10 1 8 resnet56 hetero 100 20 64 0.001 cifar100 "./../../../data/cifar100" adam 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 1 8 resnet56 hetero 100 20 64 0.001 cifar100 "./../../../data/cifar100" adam 0 > ./fedavg-resnet-hetero-cifar100.txt 2>&1 &
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
sh run_fedavg_distributed_pytorch.sh 10 10 1 8 mobilenet homo 100 20 64 0.001 cifar10 "./../../../data/cifar10" adam 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 1 8 mobilenet homo 100 20 64 0.001 cifar10 "./../../../data/cifar10" adam 0 > ./fedavg-mobilenet-homo-cifar10.txt 2>&1 &
```

train on non-IID dataset
```
sh run_fedavg_distributed_pytorch.sh 10 10 1 8 mobilenet hetero 100 20 64 0.001 cifar10 "./../../../data/cifar10" adam 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 1 8 mobilenet hetero 100 20 64 0.001 cifar10 "./../../../data/cifar10" adam 0 > ./fedavg-mobilenet-hetero-cifar10.txt 2>&1 &
```


#### CIFAR100
train on IID dataset 
```
sh run_fedavg_distributed_pytorch.sh 10 10 1 8 mobilenet homo 100 20 64 0.001 cifar100 "./../../../data/cifar100" adam 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 1 8 mobilenet homo 100 20 64 0.001 cifar100 "./../../../data/cifar100" adam 0 > ./fedavg-mobilenet-homo-cifar100.txt 2>&1 &
```

train on non-IID dataset
```
sh run_fedavg_distributed_pytorch.sh 10 10 1 8 mobilenet hetero 100 20 64 0.001 cifar100 "./../../../data/cifar100" adam 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 1 8 mobilenet hetero 100 20 64 0.001 cifar100 "./../../../data/cifar100" adam 0 > ./fedavg-mobilenet-hetero-cifar100.txt 2>&1 &
```


#### CINIC10
train on IID dataset 
```
sh run_fedavg_distributed_pytorch.sh 10 1 8 mobilenet homo 100 20 64 0.001 cinic10 "./../../../data/cinic10" adam 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 1 8 mobilenet homo 100 20 64 0.001 cinic10 "./../../../data/cinic10" adam 0 > ./fedavg-mobilenet-homo-cinic10.txt 2>&1 &
```

train on non-IID dataset
```
sh run_fedavg_distributed_pytorch.sh 10 1 8 mobilenet hetero 100 20 64 0.001 cinic10 "./../../../data/cinic10" adam 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 1 8 mobilenet hetero 100 20 64 0.001 cinic10 "./../../../data/cinic10" adam 0 > ./fedavg-mobilenet-hetero-cinic10.txt 2>&1 &
```


sh run_fedavg_distributed_pytorch.sh 10 10 1 4 lr hetero 200 20 10 0.03 mnist "./../../../data/mnist" 0
#### ImageNet -- ILSVRC2012
```
CLIENT_NUM=$1
WORKER_NUM=$2
SERVER_NUM=$3
GPU_NUM_PER_SERVER=$4
MODEL=$5
DISTRIBUTION=$6
ROUND=$7
EPOCH=$8
BATCH_SIZE=$9
LR=$10
DATASET=$11
DATA_DIR=$12
CI=$13
```
train on non-IID dataset
```
# 100 clients
sh run_fedavg_distributed_pytorch.sh 100 2 1 2 mobilenet hetero 100 1 32 0.001 ILSVRC2012 "~/datasets/landmarks/cache" 0
sh run_fedavg_distributed_pytorch.sh 100 2 1 2 mobilenet hetero 100 1 32 0.001 ILSVRC2012 "your_data_dir" 0
# 1000 clients
sh run_fedavg_distributed_pytorch.sh 1000 2 1 2 mobilenet hetero 100 1 32 0.001 ILSVRC2012 "your_data_dir" 0
```
#### gld23k
train on non-IID dataset
```
sh run_fedavg_distributed_pytorch.sh 233 2 1 2 mobilenet hetero 100 1 32 0.001 gld23k "~/datasets/landmarks" 0
sh run_fedavg_distributed_pytorch.sh 233 2 1 2 mobilenet hetero 100 1 32 0.001 gld23k "your_data_dir" 0
```

#### gld160k
train on non-IID dataset
```
sh run_fedavg_distributed_pytorch.sh 1262 2 1 2 mobilenet hetero 100 1 32 0.001 gld160k "your_data_dir" 0
```






