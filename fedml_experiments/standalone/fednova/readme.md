## Experimental Tracking Platform (report real-time result to wandb.com, please change ID to your own)
wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408


## Experiment Scripts
Heterogeneous distribution (Non-IID) experiment:

Frond-end debugging:
``` 
## Sythetic
sh run_fednova_standalone_pytorch.sh 0 10 10 10 synthetic_1_1 ./../../../data/synthetic_1_1 lr hetero 200 20 0.03 0 0 0 0 0 0 0 0

## CIFAR10
sh run_fednova_standalone_pytorch.sh 0 16 10 10 cifar10 ./../../../data/cifar10 vgg hetero 200 20 0.03 0 0 0 0 0 0 0 0

```

Please make sure to run on the background when you start training after debugging. An example to run on the background:
``` 
# MNIST
nohup sh run_fednova_standalone_pytorch.sh 0 10 10 10 mnist ./../../../data/mnist lr hetero 200 20 0.03 0 0 0 0 0 0 0 0 > ./fednova_standalone.txt 2>&1 &
```
