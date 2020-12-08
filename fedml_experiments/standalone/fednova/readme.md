## Experimental Tracking Platform (report real-time result to wandb.com, please change ID to your own)
wandb login ${YOUR_WANDB_KEY}


## Experiment Scripts
Heterogeneous distribution (Non-IID) experiment:

## Sythetic
```
sh run_fednova_standalone_pytorch.sh 0 10 10 10 synthetic_1_1 ./../../../data/synthetic_1_1 lr hetero 200 20 0.03 0 0 0 0 0 0 0 0
```

## CIFAR10
```
sh run_fednova_standalone_pytorch.sh 0 10 10 10 cifar10 ./../../../data/cifar10 vgg hetero 200 20 0.001 0 0 0 0 0 0 0 0
```
The experiment result refers to :https://wandb.ai/elliebababa/fedml/runs/51thqi89

