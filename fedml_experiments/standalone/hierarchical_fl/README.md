# Hierarchical Federated Learning
Introduction:
Provided Implementations:

## Experimental Tracking Platform (report real-time result to wandb.com, please change ID to your own)
wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408

## Experiment Scripts

### Parameter
'run_standalone_pytorch.sh' takes as input the same 10 parameters as 'fedml_experiments/standalone/fedavg' does as well as 5 additional group-related parameters at the end.
```
--group_method : how clients should be grouped 
--group_num : the number of groups
--global_comm_round : the number of global communications
--group_comm_round : the number of group communications within a global communication interval
--epochs : the number of epochs in a client within a group interval
```

### Benchmark
```
# group_method=random & group_num=10

# global_comm_round=1 & group_comm_round=10 & epochs=50
sh run_standalone_pytorch.sh 0 1000 100 10 mnist ./../../../data/mnist lr hetero 0.03 sgd random 10 1 10 50

# global_comm_round=5 & group_comm_round=2 & epochs=50
sh run_standalone_pytorch.sh 0 1000 100 10 mnist ./../../../data/mnist lr hetero 0.03 sgd random 10 5 2 50

# global_comm_round=10 & group_comm_round=1 & epochs=50
sh run_standalone_pytorch.sh 0 1000 100 10 mnist ./../../../data/mnist lr hetero 0.03 sgd random 10 10 1 50

# global_comm_round=10 & group_comm_round=5 & epochs=10
sh run_standalone_pytorch.sh 0 1000 100 10 mnist ./../../../data/mnist lr hetero 0.03 sgd random 10 10 5 10

# global_comm_round=10 & group_comm_round=50 & epochs=1
sh run_standalone_pytorch.sh 0 1000 100 10 mnist ./../../../data/mnist lr hetero 0.03 sgd random 10 10 50 1
```

![benchmark](/docs/image/hierarchical_fl_benchmark.png)

For large DNNs (ResNet, Transformer, etc), please use the distributed computing (fedml_api/distributed). 



### Benchmark Results
We publish benchmark experimental results at wanb.com: \
https://app.wandb.ai/automl/fedml
