# M1 冻结参数配方 (EXPERIMENT_PARAMS.md)

> 冻结时间：2026-04-01 08:23:40

> 一旦 M1 达标即冻结，M2/M3/M4 不得修改以下参数


## 全局固定参数

```yaml
local_epochs: 1
batch_size: 64
client_optimizer: sgd
momentum: 0.9
server_lr: 1.0        # FedAvg η_g = 1.0
server_momentum: 0.0   # 纯 FedAvg 无服务器动量
client_num_in_total: 10
client_num_per_round: 10  # 全量参与
attack_type: None     # M1 无攻击
defense_type: None    # M1 无防御
seeds: [0, 1, 2]      # 基础; α=0.1 CIFAR-10 含 [3, 4]
```

## 任务线差异参数

| 参数 | CIFAR-10 + ResNet-18 | MNIST + LeNet-5 |
|:-----|:---------------------|:----------------|
| learning_rate (η_l) | 0.01 | 0.01 |
| weight_decay | 1e-4 | 0 |
| comm_round (T) | 100 | 50 |
| Dirichlet α 集合 | {0.1, 0.3, 0.5, 100} | {0.1, 0.3, 0.5, 100} |

## 校准记录

无校准。使用推荐默认配方 E=1, η_l=0.01。


## 实验环境声明 (D4)

- **GPU**: 未知
- **CUDA**: 未知
- **PyTorch**: 2.6.0+cu124
- **Python**: 3.10
- **通信后端**: MPI (OpenMPI)
- **进程数**: 11 (1 server + 10 clients)