## Single process training
This is built for single process training to make sure your model and dataset can work well when single process training


## Run Experiments
Note please tune hyper-parameters accordingly. 
You can refer the hyper-parameter suggestions at `FedML/benchmark/README.md`), but this may not be the optimal.

Parameters:
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
LR=${10}
DATASET=${11}
DATA_DIR=${12}
CLIENT_OPTIMIZER=${13}
CI=${14}
PYTHON=${15}
```

#### ImageNet -- ILSVRC2012

```
sh run_single.sh 100 0 0 0 mobilenet_v3 hetero 100 1 32 0.1 ILSVRC2012 your_data_path sgd 0 your_python_path
```




#### gld23k

```
sh run_single.sh 100 0 0 0 mobilenet_v3 hetero 100 1 32 0.1 gld23k your_data_path sgd 0 your_python_path
```



#### gld160k

```
sh run_single.sh 100 0 0 0 mobilenet_v3 hetero 100 1 32 0.1 gld160k your_data_path sgd 0 your_python_path
```


















