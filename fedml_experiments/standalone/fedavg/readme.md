## Experimental Tracking Platform (report real-time result to wandb.com)
wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408

## Experiment Scripts
1. Homogeneous distribution (IID) experiment:
``` 
# CIFAR10, ResNet56
nohup sh run_fedavg_standalone_pytorch.sh 2 cifar10 ./../../../data/cifar10 resnet56 homo 200 20 0.001 > ./fedavg_standalone.txt 2>&1 &

# CIFAR10, MobileNet
nohup sh run_fedavg_standalone_pytorch.sh 3 cifar10 ./../../../data/cifar10 mobilenet homo 200 20 0.001 > ./fedavg_standalone.txt 2>&1 &

# CIFAR100, ResNet56
nohup sh run_fedavg_standalone_pytorch.sh 4 cifar100 ./../../../data/cifar100 resnet56 homo 200 20 0.001 > ./fedavg_standalone.txt 2>&1 &

# CIFAR100, MobileNet
nohup sh run_fedavg_standalone_pytorch.sh 5 cifar100 ./../../../data/cifar100 mobilenet homo 200 20 0.001 > ./fedavg_standalone.txt 2>&1 &

# CINIC10, ResNet56
nohup sh run_fedavg_standalone_pytorch.sh 6 cinic10 ./../../../data/cinic10 resnet56 homo 200 20 0.001 > ./fedavg_standalone.txt 2>&1 &

# CINIC10, MobileNet
nohup sh run_fedavg_standalone_pytorch.sh 7 cinic10 ./../../../data/cinic10 mobilenet homo 200 20 0.001 > ./fedavg_standalone.txt 2>&1 &
```


2. Heterogeneous distribution (Non-IID) experiment:
``` 
# CIFAR10, ResNet56
nohup sh run_fedavg_standalone_pytorch.sh 2 cifar10 ./../../../data/cifar10 resnet56 hetero 200 20 0.001 > ./fedavg_standalone.txt 2>&1 &

# CIFAR10, MobileNet
nohup sh run_fedavg_standalone_pytorch.sh 3 cifar10 ./../../../data/cifar10 mobilenet hetero 200 20 0.001 > ./fedavg_standalone.txt 2>&1 &

# CIFAR100, ResNet56
nohup sh run_fedavg_standalone_pytorch.sh 4 cifar100 ./../../../data/cifar100 resnet56 hetero 200 20 0.001 > ./fedavg_standalone.txt 2>&1 &

# CIFAR100, MobileNet
nohup sh run_fedavg_standalone_pytorch.sh 5 cifar100 ./../../../data/cifar100 mobilenet hetero 200 20 0.001 > ./fedavg_standalone.txt 2>&1 &

# CINIC10, ResNet56
nohup sh run_fedavg_standalone_pytorch.sh 6 cinic10 ./../../../data/cinic10 resnet56 hetero 200 20 0.001 > ./fedavg_standalone.txt 2>&1 &

# CINIC10, MobileNet
nohup sh run_fedavg_standalone_pytorch.sh 7 cinic10 ./../../../data/cinic10 mobilenet hetero 200 20 0.001 > ./fedavg_standalone.txt 2>&1 &
```


### Benchmark Results
We publish benchmark experimental results at wanb.com: \
https://app.wandb.ai/automl/fedml
