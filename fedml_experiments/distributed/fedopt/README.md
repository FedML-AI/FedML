## Installation
http://doc.fedml.ai/#/installation-distributed-computing

## Run Experiments
Note please tune hyper-parameters accordingly. 
You can refer the hyper-parameter suggestions at `FedML/benchmark/README.md`), but this may not be the optimal.


# Federated EMNIST experiments
```
sh run_fedopt_distributed_pytorch.sh 10 10 1 4 cnn 100 1 20 0.1 femnist "./../../../data/FederatedEMNIST/datasets" sgd 0

```
