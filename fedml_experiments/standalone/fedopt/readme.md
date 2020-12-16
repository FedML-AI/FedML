## Experimental Tracking Platform 
(report real-time result to wandb.com, please change ID to your own)
wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408


## Experiments 

#### MNIST
```
sh run_fedopt_standalone_pytorch.sh 0 10 10 10 mnist ./../../../data/mnist lr hetero 200 20 0.03 sgd 0.008 adam 0
```
#### shakespeare (LEAF)
```
sh run_fedopt_standalone_pytorch.sh 0 10 10 10 shakespeare ./../../../data/shakespeare rnn hetero 100 1 0.8 sgd 0.008 adam 0
```
#### fed_shakespeare (Google)
```
sh run_fedopt_standalone_pytorch.sh 0 10 10 4 fed_shakespeare ./../../../data/fed_shakespeare rnn hetero 1500 10 1 sgd 1 sgd 0
```
#### Federated EMNIST
 Note: You may need gradient clipping to get training run on Fed EMNIST dataset successfully. To do this, please uncomment the line "torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)" in FedML/fedml_api/standalone/fedopt/client.py.
```
sh run_fedopt_standalone_pytorch.sh 0 10 10 20 femnist ./../../../data/FederatedEMNIST/datasets cnn hetero 1500 10 0.1 sgd 1 sgd 0
```

#### Fed_CIFAR100
```
sh run_fedopt_standalone_pytorch.sh 0 10 10 10 fed_cifar100 ./../../../data/fed_cifar100/datasets resnet18_gn hetero 200 1 0.3 sgd 1 sgd 0
```
#### Stackoverflow_LR
```
sh run_fedopt_standalone_pytorch.sh 0 10 10 10 stackoverflow_lr ./../../../data/stackoverflow lr hetero 200 1 0.03 sgd 0.008 adam 0
```

#### Stackoverflow_NWP
```
sh run_fedopt_standalone_pytorch.sh 0 10 10 10 stackoverflow_nwp ./../../../data/stackoverflow rnn hetero 200 1 0.03 sgd 0.008 adam 0
```

### Results
| Dataset | Model | Accuracy |
| ------- | ------ | ------- |
| MNIST | cnn | 0.81 |
| fed_shakespeare (Google) | rnn | 0.49 |
| Federated EMNIST | cnn | 0.82 |
