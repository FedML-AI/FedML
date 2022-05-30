## Installation
http://doc.fedml.ai/#/installation-distributed-computing

## Run Experiments
Note please tune hyper-parameters accordingly. 
You can refer the hyper-parameter suggestions at `FedML/benchmark/README.md`), but this may not be the optimal.

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
sh run_fedavg_distributed_pytorch.sh 100 2 1 2 mobilenet_v3 hetero 100 1 32 "0.1" ILSVRC2012 "~/datasets/landmarks/cache" adam 0 "local_gpu_util.yaml" mpi_host_file ~/anaconda3/envs/py36/bin/python


sh run_fedavg_distributed_pytorch.sh 100 2 1 2 efficientnet hetero 100 1 32 0.1 ILSVRC2012 "your_data_dir" adam 0
# 1000 clients
sh run_fedavg_distributed_pytorch.sh 1000 2 1 2 mobilenet_v3 hetero 100 1 32 0.1 ILSVRC2012 "your_data_dir" adam 0
```
#### gld23k
train on non-IID dataset
```
sh run_fedavg_distributed_pytorch.sh 233 4 1 4 mobilenet_v3 hetero 100 1 32 0.1 gld23k "../../../../fedml.ai/data/gld/" adam 0
```

#### gld160k
train on non-IID dataset
```
sh run_fedavg_distributed_pytorch.sh 1262 2 1 2 mobilenet_v3 hetero 100 1 32 0.1 gld160k "your_data_dir" adam 0
```






