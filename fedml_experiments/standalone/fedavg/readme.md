## Experimental Tracking Platform (report real-time result to wandb.com)
wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408

## Experiment Scripts
1. Homogeneous distribution (IID) experiment:
``` 
nohup sh run_fedavg_standalone_pytorch.sh 0 cifar10 ./../../../data/cifar10 resnet56 homo 100 20 0.01 > ./log/fedavg_standalone.txt 2>&1 &
```


2. Heterogeneous distribution (Non-IID) experiment:
```
nohup sh run_fedavg_standalone_pytorch.sh 0 cifar10 ./../../../data/cifar10 resnet56 hetero 100 20 0.01 > ./log/fedavg_standalone.txt 2>&1 &
```

### Benchmark Results
We publish benchmark experimental results at wanb.com: \
https://app.wandb.ai/automl/fedml
