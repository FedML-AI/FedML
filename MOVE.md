# ShieldFL → FedML 迁移方案

> 目标：将 `ShieldFL/` 中原有自研联邦学习鲁棒聚合框架的**核心实现**优雅迁移到 FedML，最大化复用 FedML 的：
>
> - 联邦训练执行框架（**以 cross-silo / cross-cloud 为主**；simulation 仅复用目录/脚手架与部分流程）
> - 客户端/服务端生命周期与通信后端
> - 安全攻击与防御实验框架
> - benchmark / baseline 工程
> - 配置、日志、MLOps、部署与硬件编排能力
>
> 同时避免继续维护一套独立的 Flower/Ray 仿真外壳。

---

## 1. 总体判断

**建议迁移，而且建议“保留算法核心、替换框架外壳”。**

并且需要补充一个重要判断：

- `ShieldFL` 原有的“GPU 拆分”能力**可以迁到 FedML**，但应准确理解为：
   - **客户端级 / 进程级 GPU 资源并行**
   - 加上服务端 `GPUAccelerator` 的**单设备矩阵化 fitness 加速**
   - **不是** 单个 client 内部的 `DataParallel` / `DistributedDataParallel` 多卡同步训练
- FedML **支持单 GPU 乃至纯 CPU 的迁移有效性验证**
- 若要做到“同参数 + 同随机种子下必须产生相同结果”，需要在 FedML 默认随机种子控制之上，再补一层**显式确定性约束**（如固定 client 采样顺序、关闭 `cudnn.benchmark`、必要时启用 `torch.use_deterministic_algorithms(True)`）

同时，基于对当前 FedML 仓库源码的核验，需要明确几个**现实约束**：

- FedML 的 `ServerAggregator.aggregate()` **并不是在所有运行时都会真正被调用**：
   - `cross-silo horizontal` / `cross-cloud`：**可行**，会真实调用 `on_before_aggregation()` / `aggregate()` / `on_after_aggregation()`
   - `simulation + sp`：**不适合作为 VeriFL-v16 的首个 MVP 运行时**，因为当前单进程仿真器会直接走内置训练 API，忽略用户传入的自定义聚合器
   - `simulation + MPI`：**不适合作为“忠实聚合验证”运行时**，因为服务器侧聚合由内置 `FedAVGAggregator` 主导，自定义 `ServerAggregator.aggregate()` 不能作为主聚合路径生效
- 因此，**第一个真正忠实的 VeriFL-v16 迁移 MVP，应优先落在 `cross-silo horizontal`（可先用 MPI backend），其次是 `cross-cloud`；而不是 `simulation`。**
- `python/examples/federate/prebuilt_jobs/fedcv/image_classification/` 这类目录**适合作为工程骨架参考**，但**不能被当作“自定义聚合在 simulation 中已被证实可用”的证据**。
- 当前仓库的 `fedml.__init__._init_cross_silo_hierarchical()` 中残留了 `exit()`，因此 **hierarchical cross-silo 不是第一阶段的稳妥目标**。
- FedML 默认 dataset tuple 只覆盖 `train/test/local splits`，**不原生承载 ShieldFL 的 `val/trust` 四分割语义**；这部分必须由 ShieldFL 自定义数据层继续维护。

`ShieldFL/` 的核心创新主要集中在：

- `src/strategies/ours/v16.py`：VeriFL-v16 / 三阶段鲁棒聚合
- `src/strategies/ours/ga_base.py`：Micro-GA 基座
- `src/strategies/ours/gpu_accelerator.py`：GPU 向量化适应度评估与 BN 校准
- `src/attacks/*`：攻击实现（label flip / backdoor / scaling / pure scaling / byzantine / model replacement）
- `src/core/evaluator.py`：验证集 fitness、封存 test、ASR、trust 评估协议
- `src/factories/data_factory.py`：训练/验证/可信/测试四分割与 Dirichlet Non-IID 划分

而 `ShieldFL/` 中比较“重但不该继续背着跑”的部分是：

- `src/core/runner.py`：Flower + Ray 仿真编排壳
- `src/clients.py`：Flower `NumPyClient` 适配层
- `src/utils/config.py`：自有 YAML 解析与对象拼装
- `src/utils/monitor.py` / `io.py`：结果追踪与落盘外壳
- `src/core/strategy_builder.py`：Flower strategy 装配层

FedML 恰好在这些“外壳层”已经提供了更完整的替代品。因此推荐的总策略是：

> **保留“算法与评测核心”，废弃“Flower/Ray 运行壳”，优先将核心能力重挂到 FedML 的 `ClientTrainer` / `ServerAggregator` / `FedMLRunner`（以 `cross-silo horizontal` / `cross-cloud` 为首期宿主）之上。**

---

## 2. 迁移原则

## 2.1 保留什么

以下内容应尽量原样保留，仅做接口适配：

1. **VeriFL-v16 三阶段聚合逻辑**
   - GA 零阶搜索
   - 锚点归一化投影
   - 服务器动量平滑
   - BN running stats 校准

2. **Micro-GA 基础设施**
   - `_init_population`
   - `_tournament_selection`
   - `_crossover`
   - `_mutation`
   - `calculate_fitness`

3. **GPUAccelerator**
   - 客户端参数矩阵化
   - GPU 前向 fitness 评估
   - CPU fallback 验证路径
   - trainable mask 逻辑
   - BN 校准

4. **攻击实现**
   - LabelFlip
   - Backdoor / AdaptiveBackdoor
   - Scaling / PureScaling
   - Byzantine
   - ModelReplacementBackdoor

5. **实验协议**
   - 服务器验证集 / 可信集 / 客户端池 / 封存测试集的互斥划分
   - ASR 计算协议
   - 多随机种子重复实验

## 2.2 替换什么

以下内容应尽量由 FedML 承接，不再继续自研：

1. `SimulationRunner` → `FedMLRunner`
2. Flower strategy 层 → FedML `ServerAggregator`
3. Flower client 层 → FedML `ClientTrainer`
4. 自有通信/资源编排 → FedML `cross_silo` / `cross_cloud`（`simulation` 仅在修补或自定义 algorithm flow 后再作为忠实验证路径）
5. 自有结果追踪 → FedML logging / wandb / mlops
6. 自有 benchmark 外壳 → FedML `examples/federate/*` 与 `prebuilt_jobs/*`

## 2.3 迁移要求

除算法迁移本身外，还应把以下两类能力视为**一等迁移目标**：

1. **多 GPU 并发训练能力**
    - 保留 `ShieldFL` 原有“多 client 并发 + GPU 资源切分”的实验能力
    - 在 FedML 中统一改为：
       - `fedml.device.get_device(args)` 负责设备解析
       - `gpu_mapping.yaml` / `gpu_mapping_key` 负责 worker-to-GPU 映射
       - 代码中不再出现 `cuda:0`、`torch.device("cuda")` 之类的硬编码路径

2. **可复现验证能力**
    - 必须支持：
       - 纯 CPU smoke / regression test
       - 单 GPU 数值对齐验证
       - 多 GPU 性能验证
    - 迁移后的工程要显式区分：
       - **功能正确性验证**：指标一致或在小容差内一致
       - **严格确定性验证**：同 seed、同 client 顺序、同聚合输入顺序下结果完全复现

## 2.4 运行时可行性矩阵（已核验）

下表用于约束第一阶段迁移目标，避免把“目录能放”误写成“运行时真能跑算法核心”。

| 运行时 | 自定义 `ClientTrainer` | 自定义 `ServerAggregator.aggregate()` | 适合作为 VeriFL-v16 首期 MVP | 备注 |
|---|---:|---:|---:|---|
| `simulation + sp` | 部分可借鉴 | 否 | 否 | 当前单进程仿真走内置训练 API，不能作为忠实聚合验证宿主 |
| `simulation + MPI` | 是 | 否（主聚合仍由内置 FedAvg 聚合器执行） | 否 | 可参考通信流程，但不适合验证 VeriFL 三阶段聚合 |
| `cross-silo horizontal` | 是 | 是 | **是（首选）** | 可先用 MPI backend 跑最小 MVP |
| `cross-cloud` | 是 | 是 | 是 | 作为第二优先级分布式宿主 |
| `cross-silo hierarchical` | 理论上部分支持 | 理论上可扩展 | 暂不建议 | 当前仓库初始化路径存在 `exit()`，先不要把它写成近期目标 |

---

## 3. 推荐迁移后的整体分布

为了既保留算法独立性，又不污染 FedML 核心库，推荐采用“**示例工程 + 可复用公共模块**”的双层布局。

## 3.1 最终落位建议

### A. 算法实验主工程（首选）

放在：

- `python/examples/federate/prebuilt_jobs/shieldfl/`

原因：

- `ShieldFL/` 的目标并不只是“自定义聚合器 demo”，而是完整算法、攻击、评测、benchmark 工作流
- `prebuilt_jobs` 正是 FedML 用来承载一整套联邦应用/研究工程的位置
- 便于后续和 `fedcv` / `fednlp` / `healthcare` 并列维护

建议内部结构：

- `python/examples/federate/prebuilt_jobs/shieldfl/main_fedml_shieldfl.py`
- `python/examples/federate/prebuilt_jobs/shieldfl/trainer/verifl_trainer.py`
- `python/examples/federate/prebuilt_jobs/shieldfl/trainer/verifl_aggregator.py`
- `python/examples/federate/prebuilt_jobs/shieldfl/trainer/micro_ga_base.py`
- `python/examples/federate/prebuilt_jobs/shieldfl/trainer/gpu_accelerator.py`
- `python/examples/federate/prebuilt_jobs/shieldfl/attacks/*`
- `python/examples/federate/prebuilt_jobs/shieldfl/data/*`
- `python/examples/federate/prebuilt_jobs/shieldfl/model/*`
- `python/examples/federate/prebuilt_jobs/shieldfl/eval/*`
- `python/examples/federate/prebuilt_jobs/shieldfl/config/*`

### B. 攻防对照实验入口（推荐额外挂一个）

放在：

- `python/examples/federate/security/shieldfl_verifl/`

原因：

- 方便直接对接 FedMLSecurity 的现有攻击/防御实验目录
- 便于做 VeriFL-v16 vs Krum / Foolsgold / RFA / Median / TrimmedMean 的对照
- 适合作为安全 benchmark 的统一入口

### C. 若后续需要更底层复用，再抽公共模块

如果后续发现 `VeriFLAggregator`、`GPUAccelerator`、攻击实现需要被多个示例共用，可以再抽到：

- `python/fedml/core/security/attack/shieldfl_*`（仅当攻击实现要并入 FedMLSecurity 主库时）
- `python/fedml/ml/aggregator/` 或 `python/fedml/core/alg_frame/` 的扩展目录（仅当算法正式产品化/平台化时）

**第一阶段不建议直接改 FedML 核心库。**

---

## 4. ShieldFL 模块到 FedML 的迁移映射

| ShieldFL 模块 | 当前职责 | 迁移去向 | 迁移策略 |
|---|---|---|---|
| `src/main.py` | CLI 入口 | `main_fedml_shieldfl.py` | 重写入口，改用 `fedml.init()` + `FedMLRunner` |
| `src/core/runner.py` | Flower/Ray 仿真编排 | **删除职责**，交给 `FedMLRunner` | 不迁移，只保留思路 |
| `src/clients.py` | Flower `NumPyClient` | `ClientTrainer` 子类 | 改写为 FedML 客户端训练器 |
| `src/core/evaluator.py` | val/test/ASR/trust 评估 | `trainer/verifl_aggregator.py` + `eval/metrics.py` | 部分下沉到聚合器，部分保留为工具模块 |
| `src/core/strategy_builder.py` | Flower strategy 工厂 | **删除职责** | 用 FedML 直接实例化 trainer/aggregator |
| `src/strategies/ours/v16.py` | 核心聚合逻辑 | `trainer/verifl_aggregator.py` | 保留算法，改成 `ServerAggregator` 实现 |
| `src/strategies/ours/ga_base.py` | GA 基座 | `trainer/micro_ga_base.py` | 保留核心逻辑，去 Flower 依赖 |
| `src/strategies/ours/gpu_accelerator.py` | GPU fitness/BN | `trainer/gpu_accelerator.py` | 基本原样迁移 |
| `src/attacks/*` | 攻击实现 | `attacks/*` | 保留并适配 FedML `ClientTrainer` / 安全框架 |
| `src/factories/data_factory.py` | 数据四分割 + Non-IID | `data/data_loader.py` | 保留协议，改造成 FedML 数据入口 |
| `src/factories/model_factory.py` | 模型工厂 | `model/model_hub.py` | 轻量重封装 |
| `src/utils/config.py` | 自有 YAML 配置 | `config/fedml_config.yaml` + 自定义扩展字段 | 迁移到 FedML YAML |
| `src/utils/monitor.py` | metrics.csv 跟踪 | `wandb/mlops` + 可选 CSV 回调 | 减配重写 |
| `src/utils/io.py` | checkpoint | `train/` 或 `utils/checkpoint.py` | 视需要保留 |

---

## 5. 目标架构（迁移后）

推荐的迁移后执行流如下：

```text
main_fedml_shieldfl.py
  ├── fedml.init()
  ├── fedml.device.get_device(args)
   ├── custom load_data(args) -> dataset
   │     ├── FedML 标准 tuple：train/test/local partitions
   │     └── ShieldFL 扩展资产：val/trust/eval bundles
   ├── runtime profile
   │     ├── cpu-deterministic
   │     ├── single-gpu-deterministic
   │     └── multi-gpu-throughput
  ├── create_model(args)
  ├── VeriFLTrainer(model, args)
   ├── VeriFLAggregator(model, args, extra_eval_assets)
  └── FedMLRunner(args, device, dataset, model, trainer, aggregator).run()

FedML 运行期：
  ├── ClientTrainer.train()
  │     ├── benign local SGD
  │     └── 若为恶意客户端则执行 ShieldFL attack adapter
  ├── ServerAggregator.on_before_aggregation()
  │     ├── （可选）对接 FedMLAttacker/FedMLDefender/DP
  │     └── 收集 raw client updates
  ├── ServerAggregator.aggregate()
  │     ├── Phase 1: GA search
  │     ├── Phase 2: anchor projection
  │     ├── Phase 3: server momentum
  │     └── BN recalibration
  ├── ServerAggregator.on_after_aggregation()
  └── ServerAggregator.test() / test_all()
        ├── val/test accuracy
        ├── ASR
        └── trust-related metrics
```

## 5.1 设备编排与验证模式

迁移后的 `ShieldFL` 不应只有“一套默认跑法”，而应内建三种运行档位：

### A. `cpu-deterministic`

用途：

- 最严格的迁移正确性基准
- 短轮次 smoke test / regression test
- 对齐原版 `ShieldFL` 的关键中间变量

建议配置：

- `using_gpu: false`
- `num_workers: 0`（或显式固定 worker seed）
- 固定 client 采样顺序
- 聚合前按 `client_id` 排序 updates
- `torch.use_deterministic_algorithms(True)`（若涉及不支持算子，则记录例外）

### B. `single-gpu-deterministic`

用途：

- 验证迁移版与原算法在 GPU 环境下的数值对齐
- 保留 `GPUAccelerator` 的主要性能路径

建议配置：

- `using_gpu: true`
- `gpu_id: 0` 或单卡 `gpu_mapping.yaml`
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- 必要时启用 `torch.use_deterministic_algorithms(True)`

### C. `multi-gpu-throughput`

用途：

- 复现原 `ShieldFL` 的多 client 并行实验能力
- 检查资源利用率、吞吐和收敛趋势

建议配置：

- `using_gpu: true`
- 通过 `gpu_mapping.yaml` 做 worker / process 到 GPU 的映射
- `VeriFLTrainer` 与 `VeriFLAggregator` 内部统一只使用 FedML 传入 device
- 不把该模式作为 bitwise 一致性基准，而作为性能与稳定性基准

## 5.2 对原“GPU 拆分”能力的精确定义

迁移设计中必须明确：

- 原 `ShieldFL` 的 GPU 能力主要来自 **Flower + Ray 的客户端级资源并发**
- 服务端 `GPUAccelerator` 属于 **单设备聚合评估优化**
- 当前原始仓库并**没有**真正的单 client 多卡 DDP / DP 基础设施

因此在 FedML 中的“等价迁移目标”应定义为：

1. 多 client 可稳定分配到多张 GPU 并发训练
2. 服务端聚合器可在指定 device 上执行 `GPUAccelerator`
3. 所有设备选择都由 FedML 配置层统一驱动

而不是把目标错误定义成“单个 client 内部自动多卡并行训练”。

---

## 6. 分阶段迁移方案

## Phase 0：冻结 ShieldFL 为参考基线（不再继续演化）

目标：保证迁移期间有可追溯“原始真值”。

操作：

1. 将 `ShieldFL/` 明确视为**只读参考实现**
2. 记录以下文件为“金标准来源”：
   - `ShieldFL/src/strategies/ours/v16.py`
   - `ShieldFL/src/strategies/ours/ga_base.py`
   - `ShieldFL/src/strategies/ours/gpu_accelerator.py`
   - `ShieldFL/src/attacks/*`
   - `ShieldFL/src/core/evaluator.py`
   - `ShieldFL/src/factories/data_factory.py`
3. 后续任何迁移 bug，都以这些文件行为为比对基准

交付物：

- `MOVE.md`（本文件）
- 一份迁移任务分解 issue 列表（建议后续补）

---

## Phase 1：先做“最小可运行迁移”（MVP）

目标：在 FedML 中先跑通 **CIFAR10 + ResNet20 + VeriFL-v16 + benign/no-attack**，并建立可复现验证链路。

**但这里的“FedML 中跑通”需要收紧定义：首个 MVP 应运行在 `cross-silo horizontal`（可先用 MPI backend）或 `cross-cloud`，而不是 `simulation`。**

## 6.1 目录建议

新增：

- `python/examples/federate/prebuilt_jobs/shieldfl/`

首批文件：

- `main_fedml_shieldfl.py`
- `trainer/verifl_trainer.py`
- `trainer/verifl_aggregator.py`
- `trainer/micro_ga_base.py`
- `trainer/gpu_accelerator.py`
- `data/data_loader.py`
- `model/model_hub.py`
- `config/fedml_config.yaml`
- `config/gpu_mapping.yaml`
- `config/fedml_config_cpu.yaml`
- `config/fedml_config_single_gpu.yaml`
- `utils/runtime.py`

补充建议：

- Phase 1 的 `main_fedml_shieldfl.py` 应直接以 **cross-silo horizontal** 为第一运行目标。
- 若后续确实希望在单进程内做更快的本地调试，应把它定义为：
   - **方案 A**：单独实现一个 ShieldFL 自定义 algorithm flow / local harness
   - **方案 B**：修补 FedML `simulation` 运行时后再纳入主线
- 不要把 `prebuilt_jobs/fedcv/image_classification` 的单进程例子误当成“自定义聚合已被 simulation 支持”的证据；它更适合提供目录布局和 trainer/aggregator 写法参考。

## 6.2 具体迁移

### 6.2.1 客户端训练

来源：
- `ShieldFL/src/clients.py`
- `ShieldFL/src/attacks/manager.py`

目标：
- `trainer/verifl_trainer.py`

做法：
- 继承 `fedml.core.ClientTrainer`
- 复用 `MyModelTrainer` / `ClassificationTrainer` 模式
- `train()` 中保持原来的本地 SGD 逻辑
- 将“是否恶意、执行何种攻击”的逻辑接入 trainer 内部

建议接口：

- 良性客户端：正常训练
- 恶意客户端：训练后调用 attack adapter 对 `state_dict` 做变换

说明：
- FedML 默认 `ClientTrainer.update_dataset()` 已和攻击器有挂钩，但你的攻击实现更丰富，第一阶段建议**先不硬塞进 FedMLAttacker**，而是在 `VeriFLTrainer` 内自行分派
- 需要额外明确 client ID 语义：在 `cross-silo horizontal` 中，`TrainerDistAdapter` 会把 `client_rank - 1` 作为 `trainer.id`。若继续沿用 ShieldFL“前 `num_malicious` 个 client 为恶意”约定，应以这个稳定的 `trainer.id` / `client_index` 为准，而不是网络层 real edge ID。

### 6.2.2 服务端聚合器

来源：
- `ShieldFL/src/strategies/ours/v16.py`
- `ShieldFL/src/strategies/ours/ga_base.py`
- `ShieldFL/src/strategies/ours/gpu_accelerator.py`
- `ShieldFL/src/core/evaluator.py`

目标：
- `trainer/verifl_aggregator.py`
- `trainer/micro_ga_base.py`
- `trainer/gpu_accelerator.py`

做法：
- `VeriFLAggregator` 继承 `fedml.core.ServerAggregator`
- 把 `StrategyV16.aggregate_fit()` 的核心逻辑重写到：
  - `aggregate()` 为主
  - `on_after_aggregation()` 为辅（若要单独做 BN 校准/后处理）
- 把 `global_model_buffer` / `velocity_buffer` 变成聚合器实例状态
- 保留 `GPUAccelerator` 作为聚合器内部成员
- `GPUAccelerator` 必须支持：
   - CPU fallback
   - 指定 device 初始化
   - 在 deterministic 模式下禁用隐式设备选择
- `test()` 中整合 val/test/ASR 输出

关键点：
- FedML 的 `aggregate()` 收到的是 `List[Tuple[sample_num, OrderedDict]]`
- 你原来基于 Flower 的 `List[List[np.ndarray]]`，需要写一层 `state_dict <-> ndarray list` 转换
- 所有 client updates 在聚合前应按稳定键（如 `client_idx`）排序，避免并发执行导致输入顺序漂移
- **运行时限制必须写清楚**：上面的设计只有在 `cross-silo horizontal` / `cross-cloud` 下才能作为主聚合路径忠实生效；不要把 `simulation + sp` 或 `simulation + MPI` 作为 Phase 1 聚合验证宿主。

### 6.2.3 数据协议

来源：
- `ShieldFL/src/factories/data_factory.py`

目标：
- `data/data_loader.py`

做法：
- 保留“训练集四分割 + Dirichlet Non-IID”协议
- 输出为 FedML 需要的数据集 tuple
- 优先兼容 `FedMLRunner` 所期待的 dataset 结构

FedML 当前 cross-silo / cross-cloud 入口实际消费的标准 dataset 结构为：

`[train_data_num, test_data_num, train_data_global, test_data_global, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]`

因此建议：

- **标准 tuple 只承载 FedML 运行时真正消费的 8 个槽位**
- `server_val_loader` / `server_trust_loader` / `val_images` / `val_labels` 等 ShieldFL 扩展资产，不要强塞进 tuple，而应：
   - 作为 `extra_eval_assets` 传入 `VeriFLAggregator` / `VeriFLTrainer`
   - 或封装为自定义数据 bundle，在入口拆解后分别喂给 `dataset` 与 `aggregator`

必须保留的语义：
- `server_val_set ∩ client_pool = ∅`
- `server_trust_set ∩ client_pool = ∅`
- `server_val_set ∩ server_trust_set = ∅`
- test 集封存，不参与 GA fitness

### 6.2.4 模型入口

来源：
- `ShieldFL/src/factories/model_factory.py`
- `ShieldFL/src/env/models/*`

目标：
- `model/model_hub.py`

做法：
- 先迁 `ResNet20` 与 `SimpleCNN`
- 保持 PyTorch 实现不变
- 提供 `create_model(args)`

## 6.2.5 运行时与设备治理

来源：
- `ShieldFL/src/utils/context.py`
- `ShieldFL/src/core/runner.py`
- `ShieldFL/项目环境与计算资源依赖白皮书.md`

目标：
- `utils/runtime.py`
- `config/fedml_config_cpu.yaml`
- `config/fedml_config_single_gpu.yaml`
- `config/gpu_mapping.yaml`

做法：

- 新建统一运行时入口，负责：
   - 解析 `shieldfl_args.runtime_mode`
   - 设置 deterministic 开关
   - 校验 `using_gpu / gpu_mapping_file / gpu_mapping_key / gpu_id`
   - 打印当前 device、seed、可见 GPU、client 排序策略
- 彻底移除原始 `ShieldFL` 中的 `cuda:0` 默认语义
- 所有组件只接收上游已经解析好的 `device`

并补充一个实现事实：

- `fedml.init()` 当前会设置 `random_seed`、Python/NumPy/Torch seed，并打开 `torch.backends.cudnn.deterministic = True`
- 但 **它不会自动关闭** `torch.backends.cudnn.benchmark`
- 也 **不会自动启用** `torch.use_deterministic_algorithms(True)`

所以 deterministic 模式仍应由 ShieldFL 自己显式补齐这些约束。

说明：

- 这一步看起来像“杂活”，但实际上是迁移成败的分水岭。
- 原始 `ShieldFL` 的多 GPU 问题，本质上不是算法问题，而是**资源编排与设备一致性**问题。

---

## Phase 2：把原有攻击体系迁入 FedML 工程

目标：在 FedML 里复现 `ShieldFL/` 的攻击菜单，而不是只依赖内建攻击。

## 6.3 推荐落位

放在：

- `python/examples/federate/prebuilt_jobs/shieldfl/attacks/`

原因：
- 初期不要直接改 `python/fedml/core/security/attack/`
- 保持 ShieldFL 攻击实现自治
- 便于和论文实验协议一起维护

## 6.4 攻击迁移方式

### 第一层：保留 ShieldFL 原攻击类

直接迁入：

- `base.py`
- `manager.py`
- `label_flip.py`
- `backdoor.py`
- `scaling.py`
- `pure_scaling.py`
- `byzantine_attack.py`
- `model_replacement_backdoor_attack.py`

### 第二层：写 FedML 适配器

在 `VeriFLTrainer` 中增加：

- 恶意客户端判定
- 训练前/训练后攻击注入
- 与 `args` / FedML config 的映射

### 第三层：与 FedMLSecurity 的关系

短期：
- 作为独立 attack family，与 FedML 内建攻击并存

中期：
- 将通用攻击（如 byzantine / model replacement）与 FedML 现有实现对齐
- 把 truly custom 的攻击（如 pure_scaling / adaptive_backdoor 协议）继续保留在 ShieldFL 子工程中

---

## Phase 3：迁移评估协议与日志

目标：把 `Evaluator + MetricMonitor` 收拢到 FedML 风格，而不是完整照搬。

## 6.5 建议拆分

### 保留为工具模块

放在：

- `eval/metrics.py`
- `eval/asr.py`
- `eval/trust.py`

保留内容：
- `evaluate_test`
- `evaluate_backdoor_test`
- trust update 计算
- ASR 注入逻辑

### 下沉到聚合器

由 `VeriFLAggregator.test()` 负责：
- 在 server 侧执行测试
- 返回 FedML 约定 metrics 字典

### 日志追踪

替换建议：
- 轮级指标：走 FedML 自带 logging / wandb
- CSV 落盘：作为可选工具保留最小实现
- 不再保留 `SimulationRunner` 那种自定义 round loop 监控壳

---

## Phase 4：接入 benchmark / baseline 对照体系

目标：让 VeriFL-v16 不只是“可跑”，还能够在 FedML 生态里成为可对照的方法。

## 6.6 对照方式

### 对照一：FedML 安全目录

在：
- `python/examples/federate/security/shieldfl_verifl/`

对照：
- VeriFL-v16
- Krum
- MultiKrum
- FoolsGold
- RFA
- Coordinate-wise median
- Trimmed mean
- SLSGD
- CRFL

### 对照二：FedCV benchmark

在：
- `python/examples/federate/prebuilt_jobs/fedcv/image_classification`

建议复用：
- CIFAR10 / CIFAR100 数据工作流
- ResNet 系模型
- classification trainer 示例与目录结构；**不要直接复用其 aggregator 作为行为基准**，因为该示例更像最小骨架，并不构成 VeriFL 聚合流程的可用证明

### 对照三：privacy / secure aggregation

与：
- `python/examples/federate/privacy/*`
- `cross_silo/light_sec_agg_example`

做兼容性实验，验证：
- VeriFL-v16 + CDP/LDP
- VeriFL-v16 + secure aggregation（若可行）

---

## Phase 5：迁移部署与编排

目标：复用 FedML 的 Launch / MLOps / 多机调度，而不是维持自有 Ray 资源配置，并将多 GPU 资源调度纳入标准配置路径。

## 6.7 配置迁移

来源：
- `ShieldFL/configs/scenarios/*.yaml`
- `ShieldFL/configs/methods/*.yaml`
- `ShieldFL/src/utils/config.py`

目标：
- `config/fedml_config.yaml`
- 可选：附加 `shieldfl_config.yaml`

迁移原则：

### 直接映射到 FedML 的字段

- `dataset` → `data_args.dataset`
- `alpha` → `data_args.partition_alpha`
- `rounds` → `train_args.comm_round`
- `num_clients` → `train_args.client_num_in_total`
- `num_clients_per_round` → `train_args.client_num_per_round`
- `epochs` → `train_args.epochs`
- `batch_size` → `train_args.batch_size`
- `learning_rate` → `train_args.learning_rate`
- `using_gpu` / `gpu_mapping` → `device_args.*`

### 保留为自定义扩展字段

以下不属于标准 FedML 字段，但可安全加到 YAML 中，由自定义 trainer/aggregator 消费：

- `shieldfl_args.pop_size`
- `shieldfl_args.generations`
- `shieldfl_args.lambda_reg`
- `shieldfl_args.server_momentum`
- `shieldfl_args.server_lr`
- `shieldfl_args.server_val_size`
- `shieldfl_args.server_trust_size`
- `shieldfl_args.trigger_size`
- `shieldfl_args.attack_type`
- `shieldfl_args.attack_params`
- `shieldfl_args.num_malicious`
- `shieldfl_args.eval_asr`
- `shieldfl_args.runtime_mode`（`cpu-deterministic` / `single-gpu-deterministic` / `multi-gpu-throughput`）
- `shieldfl_args.enforce_determinism`
- `shieldfl_args.sort_client_updates`

这样可以做到：
- 运行入口走 FedML 标准配置
- 算法特有参数仍然结构化、可追踪

### 多 GPU 映射原则

迁移后必须遵守以下原则：

1. 多 GPU 能力统一通过 FedML 的 `device_args` 管理
2. worker/process 到 GPU 的映射统一放在 `config/gpu_mapping.yaml`
3. `ClientTrainer`、`ServerAggregator`、`GPUAccelerator` 内部都不直接猜测 GPU 编号
4. 若需要作业级隔离，优先结合 `CUDA_VISIBLE_DEVICES` 与 FedML mapping，而不是在代码里写死设备

---

## 7. 推荐的“优雅分布”目录蓝图

```text
/opt/dev/FedML/
├── MOVE.md
├── ShieldFL/                              # 只读参考实现，不再作为运行入口
└── python/
    └── examples/
        └── federate/
            ├── prebuilt_jobs/
            │   └── shieldfl/
            │       ├── main_fedml_shieldfl.py
            │       ├── README.md
            │       ├── config/
            │       │   ├── fedml_config.yaml
            │       │   └── gpu_mapping.yaml
            │       ├── trainer/
            │       │   ├── verifl_trainer.py
            │       │   ├── verifl_aggregator.py
            │       │   ├── micro_ga_base.py
            │       │   └── gpu_accelerator.py
            │       ├── attacks/
            │       │   ├── base.py
            │       │   ├── manager.py
            │       │   ├── label_flip.py
            │       │   ├── backdoor.py
            │       │   ├── scaling.py
            │       │   ├── pure_scaling.py
            │       │   ├── byzantine_attack.py
            │       │   └── model_replacement_backdoor_attack.py
            │       ├── data/
            │       │   ├── data_loader.py
            │       │   └── partition.py
            │       ├── model/
            │       │   ├── model_hub.py
            │       │   ├── resnet20.py
            │       │   └── simple_cnn.py
            │       ├── eval/
            │       │   ├── metrics.py
            │       │   ├── asr.py
            │       │   └── trust.py
            │       └── utils/
            │           └── checkpoint.py
            └── security/
                └── shieldfl_verifl/
                    ├── README.md
                    ├── config/
                    └── ... 对照实验入口 ...
```

---

## 8. 实施顺序（按优先级）

## P0（先做）

1. 建立 `prebuilt_jobs/shieldfl/` 目录
2. 新建 `main_fedml_shieldfl.py`（以 `cross-silo horizontal` 为首个真实运行目标）
3. 迁入 `ResNet20` / `SimpleCNN`
4. 迁入 `DataFactory` 的核心分割逻辑
5. 建立 `cpu-deterministic` 运行档位
6. 迁入 `MicroGABase` / `GPUAccelerator` / `VeriFL-v16`
7. 写 `VeriFLAggregator(ServerAggregator)`
8. 写 `VeriFLTrainer(ClientTrainer)`
9. 在无攻击场景下先跑通 3 轮 CPU smoke test
10. 固定 client 顺序与聚合顺序，验证中间结果稳定

## P1（随后）

1. 建立 `single-gpu-deterministic` 运行档位
2. 迁入 `AttackManager` 与 6 类攻击
3. 跑通：
   - label flip
   - byzantine
   - model replacement
4. 实现 ASR 评估回调
5. 验证 BN 校准与 server momentum 状态一致性
6. 在单 GPU 上与原始 `ShieldFL` 做数值回归对比

## P2（再做）

1. 建立 `multi-gpu-throughput` 运行档位
2. 补 `shieldfl_verifl` 安全实验目录
3. 与 FedML 的 security baseline 做统一对照
4. 增加 wandb / mlops 指标记录
5. 用 `gpu_mapping.yaml` 做 2-GPU / 4-GPU 并发 smoke test
6. 补 checkpoint 与 resume（若需要）

## P3（可选增强）

1. 与 FedML DP / secure aggregation 做兼容性实验
2. 将通用攻击实现逐步上收至 FedMLSecurity 兼容层
3. 若未来确有需求，再评估单 client 内的 DDP / DataParallel 支持
4. 支持 cross-cloud / Launch 编排

---

## 9. 迁移过程中的关键设计决策

## 9.1 不建议继续保留 Flower Strategy 抽象

原因：
- 这会形成“双框架嵌套”
- 后续部署、MLOps、配置和安全钩子都会别扭
- `StrategyBuilder` 会成为历史包袱

结论：
- **直接把 `StrategyV16` 的逻辑迁成 FedML `ServerAggregator`。**

## 9.2 不建议第一阶段直接修改 FedML 核心库

原因：
- 迁移初期变数太多
- 核心库修改会放大回归风险
- 示例工程足够承载算法落地

结论：
- **先落在 `python/examples/federate/prebuilt_jobs/shieldfl/`。**

补充边界：
- 如果未来确实需要把 VeriFL 放回 `simulation` 作为首选本地调试模式，再考虑最小修改 FedML 仿真器；但这不应阻塞第一阶段迁移。

## 9.3 不建议强行把所有 ShieldFL 攻击都塞进 `FedMLAttacker`

原因：
- FedML 内建攻击接口覆盖面有限
- 你的攻击中有明显论文特定协议（例如 pure_scaling / adaptive_backdoor 流程）
- 先适配 `ClientTrainer` 更快、更稳

结论：
- **短期：ShieldFL 自有攻击库 + trainer 内分派**
- **中期：挑可通用的再并入 FedMLSecurity**

## 9.4 数据协议必须保留 ShieldFL 语义，不要直接退化成 `fedml.data.load(args)` 默认行为

原因：
- VeriFL-v16 强依赖 server-side validation/trust 划分
- 这是 fitness 正确性的前提
- 默认 FedML 数据加载不会天然保证你的四分割协议

结论：
- **保留自定义 data loader，但输出对齐 FedML dataset tuple。**

---

## 10. 风险与规避策略

| 风险 | 说明 | 规避方式 |
|---|---|---|
| Flower 与 FedML 参数格式差异 | 原来用 `List[np.ndarray]`，FedML 常用 `OrderedDict` | 写统一转换层，不在算法内部散落转换代码 |
| 验证集协议被破坏 | 若直接用 FedML 默认加载，GA fitness 语义会变 | 保留自定义 data loader |
| BN running stats 异常 | 状态字典包含 buffers，错误缩放会污染模型 | 复用 `trainable_mask` 逻辑，只缩放 trainable 参数 |
| 攻击接口不一致 | 原攻击类基于自定义 `execute()` | 在 `VeriFLTrainer` 中做 adapter |
| 双重防御污染实验 | 若同时启用 FedML defense 和 VeriFL defense | VeriFL 实验默认关闭 built-in defense |
| 迁移初期跑不出同结果 | 运行壳变化会带来细微差异 | 先做 smoke test，再做数值回归验证 |
| 错误选择运行时 | 误把 `simulation` 当成忠实聚合验证宿主，会导致 VeriFL 核心聚合实际上没有真正被调用 | Phase 1 首选 `cross-silo horizontal`（可先用 MPI backend）或 `cross-cloud` |
| hierarchical 路线误判 | 当前仓库 `cross-silo hierarchical` 初始化路径残留 `exit()`，短期不可作为主线依赖 | 明确标记为后续修复项，不列入首版承诺 |
| 设备选择漂移 | 多进程并发下若仍有 `cuda:0`/自动猜卡，结果和资源占用都会失控 | 所有组件统一走 `fedml.device.get_device(args)` 与 `gpu_mapping.yaml` |
| 单 GPU / CPU 结果不稳定 | 虽然 FedML 已设置 seed，但仍可能受 cuDNN / DataLoader / 输入顺序影响 | 关闭 `cudnn.benchmark`、固定 client/update 顺序、必要时启用 deterministic algorithms |
| 错把目标理解成 DDP 迁移 | 会把范围无谓扩大、拖慢首版落地 | 明确当前迁移目标是“多 client 多 GPU 并发 + 服务端单设备加速”，不是单 client 多卡训练 |

---

## 11. 验收标准

迁移完成至少需要满足以下验收条件：

## 11.1 功能验收

- [ ] 在 FedML 中成功运行 `VeriFL-v16` 的 smoke test
- [ ] 支持 `cpu-deterministic` 运行模式
- [ ] 支持 `single-gpu-deterministic` 运行模式
- [ ] 支持 `multi-gpu-throughput` 运行模式
- [ ] 支持 benign/no-attack 基础训练
- [ ] 支持至少 3 种攻击：label_flip / byzantine / model_replacement_backdoor
- [ ] 服务端完成 GA → anchor → EMA → BN calibration 全流程
- [ ] 输出 accuracy / loss / ASR（后门场景）

## 11.2 语义验收

- [ ] `server_val_set` 与 `client_pool` 严格互斥
- [ ] `server_trust_set` 与 `client_pool` 严格互斥
- [ ] test 集仅用于评估
- [ ] BN buffers 不参与 anchor scaling
- [ ] 第一轮 EMA 初始化逻辑与 ShieldFL 一致
- [ ] 聚合前 client updates 顺序稳定可控
- [ ] 运行日志能明确打印 seed、device、runtime mode

## 11.3 实验验收

- [ ] 在同一场景下，FedML 迁移版与 `ShieldFL/` 原版的曲线趋势一致
- [ ] 在 CPU 短轮次下，同 seed 可复现关键中间结果或在严格容差内一致
- [ ] 在单 GPU 下，同 seed 可复现主要指标或在严格容差内一致
- [ ] 与 FedML 内建 baseline（如 Krum/Foolsgold/RFA）可进行统一对照
- [ ] 支持至少一种 **真正调用自定义聚合器** 的运行方式（建议先 `cross-silo horizontal + MPI backend`，再扩展到 MQTT_S3）
- [ ] 完成至少一次 2-GPU 并发 smoke test，验证多 client 到多 GPU 的映射有效

---

## 12. 推荐的首个落地里程碑（两周版本）

### Milestone 1：FedML 内最小 VeriFL-v16

交付范围：

1. `prebuilt_jobs/shieldfl/` 目录建好
2. `main_fedml_shieldfl.py` 能以 `cross-silo horizontal` 模式启动
3. `VeriFLTrainer` / `VeriFLAggregator` 可运行
4. `MicroGABase` / `GPUAccelerator` 成功接入
5. `cpu-deterministic` 配置可跑通
6. CIFAR10 + ResNet20 + no-attack + 3 rounds 跑通
7. label_flip 场景可跑
8. 每轮输出 accuracy / loss / 可选 ASR
9. 单 GPU 对齐验证入口预留完成

不强求：
- 全部攻击齐全
- Launch / MLOps 全接好
- cross-device/mobile 支持
- 单 client 级 DDP 多卡训练

---

## 13. 最终建议

如果只用一句话概括这个迁移方案：

> **把 ShieldFL 当作“算法仓”，把 FedML 当作“联邦基础设施宿主”，用 `ClientTrainer + ServerAggregator + FedMLRunner` 重挂所有核心能力。**

最优雅的分布方式不是把所有代码散着塞进 FedML 各处，而是：

1. **把 ShieldFL/VeriFL 作为一个完整 prebuilt job 落在 `python/examples/federate/prebuilt_jobs/shieldfl/`**
2. **把安全对照实验再挂一个 `python/examples/federate/security/shieldfl_verifl/` 入口**
3. **等迁移稳定后，再决定哪些通用模块值得进一步上收进 FedML 核心库**

这样可以同时做到：

- 不丢算法核心
- 不继续背自研 Flower/Ray 壳子
- 充分复用 FedML 生态
- 保持后续 benchmark / 攻防 / 部署扩展的清晰边界
