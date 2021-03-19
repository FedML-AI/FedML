## Installation
http://doc.fedml.ai/#/installation-distributed-computing

## Run Experiments
Note please tune hyper-parameters accordingly. 
You can refer the hyper-parameter suggestions at `FedML/benchmark/README.md`), but this may not be the optimal.


## Usage
```
sh run_fedavg_distributed_pytorch.sh \
    --client_num_in_total <client_num_in_total> \
    --client_num_per_round <client_num_per_round> \
    --model <model> \
    --partition_method <partition_method> 
    --comm_round <comm_round> \
    --epochs <epochs>\
    --batch_size <batch_size> \
    --learning_rate <learning_rate> \
    --dataset <dataset> \
    --data_dir <data_dir> \
    --client_optimizer <client_optimizer> \
    --backend <backend> \
    --grpc_ipconfig_path <grpc_ipconfig_path> \
    --ci <ci>
```

## Setting ip configurations for grpc
```
1. create .csv file in the format:

    receiver_id,ip
    0,<ip_0>
    ...
    n,<ip_n>
    
    where n = client_num_per_round

2. provide path to file as argument to --grpc_ipconfig_path
```
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
sh run_fedavg_distributed_pytorch.sh 10 10 1 4 cnn hetero 100 1 20 0.1 femnist "./../../../data/FederatedEMNIST/datasets" sgd sgd GRPC grpc_ipconfig.csv 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 1 4 cnn hetero 100 10 20 0.1 femnist "./../../../data/FederatedEMNIST/datasets" sgd sgd GRPC grpc_ipconfig.csv 0 > ./fedavg-cnn-femnist.txt 2>&1 &
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
sh run_fedavg_distributed_pytorch.sh 10 10 resnet56 homo 100 20 64 0.001 cifar10 "./../../../data/cifar10" adam 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 resnet56 homo 100 20 64 0.001 cifar10 "./../../../data/cifar10" adam 0 > ./fedavg-resnet-homo-cifar10.txt 2>&1 &
```

train on non-IID dataset
```
sh run_fedavg_distributed_pytorch.sh 10 10 resnet56 hetero 100 20 64 0.001 cifar10 "./../../../data/cifar10" adam 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 resnet56 hetero 100 20 64 0.001 cifar10 "./../../../data/cifar10" adam 0 > ./fedavg-resnet-hetero-cifar10.txt 2>&1 &
```


#### CIFAR100
train on IID dataset 
```
sh run_fedavg_distributed_pytorch.sh 10 10 resnet56 homo 100 20 64 0.001 cifar100 "./../../../data/cifar100" adam 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 resnet56 homo 100 20 64 0.001 cifar100 "./../../../data/cifar100" adam 0 > ./fedavg-resnet-homo-cifar100.txt 2>&1 &
```

train on non-IID dataset
```
sh run_fedavg_distributed_pytorch.sh 10 10 resnet56 hetero 100 20 64 0.001 cifar100 "./../../../data/cifar100" adam 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 resnet56 hetero 100 20 64 0.001 cifar100 "./../../../data/cifar100" adam 0 > ./fedavg-resnet-hetero-cifar100.txt 2>&1 &
```


#### CINIC10
train on IID dataset 
```
sh run_fedavg_distributed_pytorch.sh 10 10 resnet56 homo 100 20 64 0.001 cinic10 "./../../../data/cinic10" 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 resnet56 homo 100 20 64 0.001 cinic10 "./../../../data/cinic10" 0 > ./fedavg-resnet-homo-cinic10.txt 2>&1 &
```

train on non-IID dataset
```
sh run_fedavg_distributed_pytorch.sh 10 10 resnet56 hetero 100 20 64 0.001 cinic10 "./../../../data/cinic10" 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 resnet56 hetero 100 20 64 0.001 cinic10 "./../../../data/cinic10" 0 > ./fedavg-resnet-hetero-cinic10.txt 2>&1 &
```


## MobileNet Federated Training

#### CIFAR10
train on IID dataset 
```
sh run_fedavg_distributed_pytorch.sh 10 10 mobilenet homo 100 20 64 0.001 cifar10 "./../../../data/cifar10" adam 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 mobilenet homo 100 20 64 0.001 cifar10 "./../../../data/cifar10" adam 0 > ./fedavg-mobilenet-homo-cifar10.txt 2>&1 &
```

train on non-IID dataset
```
sh run_fedavg_distributed_pytorch.sh 10 10 mobilenet hetero 100 20 64 0.001 cifar10 "./../../../data/cifar10" adam 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 mobilenet hetero 100 20 64 0.001 cifar10 "./../../../data/cifar10" adam 0 > ./fedavg-mobilenet-hetero-cifar10.txt 2>&1 &
```


#### CIFAR100
train on IID dataset 
```
sh run_fedavg_distributed_pytorch.sh 10 10 mobilenet homo 100 20 64 0.001 cifar100 "./../../../data/cifar100" adam 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 mobilenet homo 100 20 64 0.001 cifar100 "./../../../data/cifar100" adam 0 > ./fedavg-mobilenet-homo-cifar100.txt 2>&1 &
```

train on non-IID dataset
```
sh run_fedavg_distributed_pytorch.sh 10 10 mobilenet hetero 100 20 64 0.001 cifar100 "./../../../data/cifar100" adam 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 mobilenet hetero 100 20 64 0.001 cifar100 "./../../../data/cifar100" adam 0 > ./fedavg-mobilenet-hetero-cifar100.txt 2>&1 &
```


#### CINIC10
train on IID dataset 
```
sh run_fedavg_distributed_pytorch.sh 10 mobilenet homo 100 20 64 0.001 cinic10 "./../../../data/cinic10" adam 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 mobilenet homo 100 20 64 0.001 cinic10 "./../../../data/cinic10" adam 0 > ./fedavg-mobilenet-homo-cinic10.txt 2>&1 &
```

train on non-IID dataset
```
sh run_fedavg_distributed_pytorch.sh 10 mobilenet hetero 100 20 64 0.001 cinic10 "./../../../data/cinic10" adam 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 mobilenet hetero 100 20 64 0.001 cinic10 "./../../../data/cinic10" adam 0 > ./fedavg-mobilenet-hetero-cinic10.txt 2>&1 &
```


sh run_fedavg_distributed_pytorch.sh 10 10 lr hetero 200 20 10 0.03 mnist "./../../../data/mnist" 0
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
CLIENT_OPTIMIZER=${13}
CI=${14}
GPU_UTIL_FILE=${15}
MPI_HOST_FILE=${16}
PYTHON=${17}

```
train on non-IID dataset
```
# 100 clients
sh run_fedavg_distributed_pytorch.sh 100 2 mobilenet_v3 hetero 100 1 32 "0.1" ILSVRC2012 "~/datasets/landmarks/cache" adam 0 "local_gpu_util.yaml" mpi_host_file ~/anaconda3/envs/py36/bin/python


sh run_fedavg_distributed_pytorch.sh 100 2 efficientnet hetero 100 1 32 0.1 ILSVRC2012 "your_data_dir" adam 0
# 1000 clients
sh run_fedavg_distributed_pytorch.sh 1000 2 mobilenet_v3 hetero 100 1 32 0.1 ILSVRC2012 "your_data_dir" adam 0
```
#### gld23k
train on non-IID dataset
```
sh run_fedavg_distributed_pytorch.sh 233 2 mobilenet_v3 hetero 100 1 32 0.1 gld23k "~/datasets/landmarks" adam 0
sh run_fedavg_distributed_pytorch.sh 233 2 mobilenet_v3 hetero 100 1 32 0.1 gld23k "your_data_dir" adam 0

sh run_fedavg_distributed_pytorch.sh 100 2 efficientnet hetero 100 1 32 "0.1" gld160k ~/datasets/landmarks adam 0 "local_gpu_util.yaml" "mpi_host_file" ~/anaconda3/envs/py36/bin/python

sh run_fedavg_distributed_pytorch.sh 100 2 efficientnet hetero 100 1 32 "0.1" gld23k "~/datasets/landmarks" adam 0 "local_gpu_util.yaml" "mpi_host_file" ~/anaconda3/envs/py36/bin/python

```

#### gld160k
train on non-IID dataset
```
sh run_fedavg_distributed_pytorch.sh 1262 2 mobilenet_v3 hetero 100 1 32 0.1 gld160k "your_data_dir" adam 0
```






