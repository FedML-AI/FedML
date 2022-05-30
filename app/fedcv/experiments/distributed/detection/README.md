## Installation
http://doc.fedml.ai/#/installation-distributed-computing

https://github.com/ultralytics/yolov5

## Run Experiments
Note please tune hyper-parameters accordingly. 
You can refer the hyper-parameter suggestions at `FedML/benchmark/README.md`), but this may not be the optimal.

#### COCO
```
CLIENT_NUM=$1
WORKER_NUM=$2
SERVER_NUM=$3
GPU_NUM_PER_SERVER=$4
DATA=$5
DISTRIBUTION=$6
ROUND=$7
EPOCH=$8
BATCH_SIZE=$9
LR=${10}
DATASET=${11}
DATA_DIR=${12}
WEIGHTS=${13}
CI=${14}
DEVICE=${15}6}
PYTHON=${17}

```
train on IID dataset (eg:coco128,4 clients)
```
sh run_fedavg_distributed_pytorch.sh 4 4 1 2 ./data/coco128.yaml homo 300 1 4 0.01 FedCV/model/detection/models/yolov5s.yaml 0 FedCV/model/detection/models/yolov5s.pt 0 0,1
```
```
```
train on Non-IID dataset (eg:coco128,4 clients)
```
sh run_fedavg_distributed_pytorch.sh 4 4 1 2 ./data/coco128.yaml hetero 300 1 4 0.01 FedCV/model/detection/models/yolov5s.yaml 0 FedCV/model/detection/models/yolov5s.pt 0 0,1
```
```





