# ShieldFL Phase 1 运行记录

## 环境

- 日期：2026-03-23
- OS：Linux
- 运行模式：FedML `cross_silo` + `MPI`
- Python：虚拟环境 `.venv`
- 数据集：本地手动下载并解压的 `CIFAR-10`

## 运行 1：`cpu_observable`

- 配置：`config/fedml_config_cpu_observable.yaml`
- 模型：`SimpleCNN`
- 进程：`1 server + 3 clients`
- 轮数：3
- GA 预算：`pop_size=15`, `generations=10`
- 设备：纯 CPU

### 关键证据

日志中确认：

- 服务端真实启动：`server client_id_list = [1, 2, 3]`
- 客户端真实训练：`#######training########### round_id = 0/1/2`
- 自定义聚合器真实执行：`VeriFL on_before_aggregation`
- Phase 1：`GA generation 1/10 ... 10/10`
- Phase 2：`VeriFL phase-2 complete | anchor_projection`
- Phase 3：`VeriFL phase-3 complete | server_momentum ...`
- 聚合完成：`VeriFL aggregate complete`

### 服务器评估结果

- round 0: `accuracy = 0.1080`, `loss = 2.304070`
- round 1: `accuracy = 0.1140`, `loss = 2.303495`
- round 2: `accuracy = 0.1160`, `loss = 2.301687`

### 判定

- 证明了 `FedML cross-silo horizontal + MPI` 宿主下，自定义 `VeriFLAggregator.aggregate()` 被真实调用
- 没有退回内置 FedAvg
- 三阶段逻辑无删减运行
- 在纯 CPU 下有可观测结果

## 运行 2：`cpu_fidelity`

- 配置：`config/fedml_config_cpu_fidelity.yaml`
- 模型：`ResNet20`
- 进程：`1 server + 2 clients`
- 轮数：1
- GA 预算：`pop_size=15`, `generations=10`
- 设备：纯 CPU

> 注：为了让 CPU fidelity checkpoint 在当前机器上可执行，仅缩小了客户端样本数与验证集大小；未缩减核心算法逻辑，也未缩减 GA 默认预算。

### 关键证据

日志中确认：

- 服务端真实聚合：`VeriFL on_before_aggregation: 2 client updates`
- Phase 1：`GA generation 1/10 ... 10/10`
- Phase 2：`VeriFL phase-2 complete | anchor_projection`
- Phase 3：`VeriFL phase-3 init | server_momentum bootstrap from first global state`
- BN 校准触发：`VeriFL bn_recalibration complete | enabled=True`
- 聚合完成：`VeriFL aggregate complete`

### 服务器评估结果

- round 0: `accuracy = 0.0859375`, `loss = 2.360559865832329`, `samples = 128`

### 判定

- `ResNet20` 的 BN 路径已在 FedML 宿主内被真实触发
- `bn_recalibration` 已执行，且日志证明 `enabled=True`
- CPU fidelity checkpoint 成功完成

## 总结

Phase 1 已完成以下核心验收：

- FedML `cross_silo` + `MPI` 宿主真实承载 ShieldFL 迁移实现
- 自定义 trainer / aggregator 被真实调用
- VeriFL-v16 三阶段聚合逻辑未降级
- `SimpleCNN` 纯 CPU 可观测运行完成
- `ResNet20` 纯 CPU BN fidelity checkpoint 完成
