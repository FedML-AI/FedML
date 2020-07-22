wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408

## Experiments
Homogeneous distribution (IID) experiment:
```
nohup sh run_fedavg_standalone_pytorch.sh 0 densenet homo 100 20 0.01 > ./log/console_log_2_1.txt 2>&1 &
```

Heterogeneous distribution (Non-IID) experiment:
```
nohup sh run_fedavg_standalone_pytorch.sh 1 densenet hetero 100 20 0.01 > ./log/console_log_4_1.txt 2>&1 &
```