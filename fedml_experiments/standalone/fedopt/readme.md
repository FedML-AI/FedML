## Experimental Tracking Platform (report real-time result to wandb.com, please change ID to your own)
wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408


## Experiment Scripts
Heterogeneous distribution (Non-IID) experiment:

Frond-end debugging:
``` 
## MNIST
sh run_fedopt_standalone_pytorch.sh 0 10 10 10 mnist ./../../../data/mnist lr hetero 200 20 0.03 sgd 0.008 adam 0

## shakespeare (LEAF)
sh run_fedopt_standalone_pytorch.sh 0 10 10 10 shakespeare ./../../../data/shakespeare rnn hetero 100 1 0.8 sgd 0.008 adam 0

# fed_shakespeare (Google)
sh run_fedopt_standalone_pytorch.sh 0 10 10 10 fed_shakespeare ./../../../data/fed_shakespeare rnn hetero 100 1 0.8 sgd 0.008 adam 0

## Federated EMNIST
## Note: You may need gradient clipping to get training run on Fed EMNIST dataset successfully. To do this, please uncomment the line "torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)" in FedML\fedml_api\standalone\fedopt\client.py.
sh run_fedopt_standalone_pytorch.sh 0 10 10 10 femnist ./../../../data/FederatedEMNIST cnn hetero 200 1 0.03 sgd 0.008 adam 0

## Fed_CIFAR100
sh run_fedopt_standalone_pytorch.sh 0 10 10 10 fed_cifar100 ./../../../data/fed_cifar100 resnet18_gn hetero 200 1 0.003 adam 0.008 adam 0
# Stackoverflow_LR
sh run_fedopt_standalone_pytorch.sh 0 10 10 10 stackoverflow_lr ./../../../data/stackoverflow lr hetero 200 1 0.03 sgd 0.008 adam 0

# Stackoverflow_NWP
sh run_fedopt_standalone_pytorch.sh 0 10 10 10 stackoverflow_nwp ./../../../data/stackoverflow rnn hetero 200 1 0.03 sgd 0.008 adam 0

# CIFAR10
sh run_fedopt_standalone_pytorch.sh 0 10 10 10 cifar10 ./../../../data/cifar10 resnet56 hetero 200 1 0.03 sgd 0.008 adam 0
```

Please make sure to run on the background when you start training after debugging. An example to run on the background:
``` 
# MNIST
nohup sh run_fedopt_standalone_pytorch.sh 2 10 10 10 mnist ./../../../data/mnist lr hetero 200 20 0.03 sgd 0 > ./fedopt_standalone.txt 2>&1 &
```

For large DNNs (ResNet, Transformer, etc), please use the distributed computing (fedml_api/distributed). 


### Benchmark Results
We publish benchmark experimental results at wanb.com: \
https://app.wandb.ai/automl/fedml
