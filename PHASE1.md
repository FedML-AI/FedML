# ShieldFL → FedML Phase 1 实施清单

> 本文只回答一件事：**如何在不对核心算法做任何逻辑降级的前提下，把 ShieldFL 的 VeriFL-v16 核心迁移到 FedML，并在纯 CPU 环境下跑出可观测结果。**
>
> Phase 1 的关键词不是“先跑起来再说”，而是：**先验证算法核心没有被框架迁移偷偷改写。**

---

## 1. Phase 1 的唯一目标

Phase 1 必须同时满足两件事：

1. **框架目标**：VeriFL-v16 的核心执行路径已经真实运行在 FedML 框架宿主中，而不是跑在旧 Flower/Ray 壳里，也不是退回内置 FedAvg。
2. **算法目标**：GA 搜索 → 锚点投影 → 服务器动量 → BN 校准 这条完整链路没有被删减、绕过、替换或弱化。

换句话说，Phase 1 不是在做“FedML 版本的近似实验”，而是在做：

> **ShieldFL 金标准算法的 FedML 忠实迁移验证。**

---

## 2. Phase 1 的硬约束（不可退让）

以下约束只要违反任意一条，Phase 1 就**不算完成**。

### 2.1 运行时约束

- 必须运行在 **FedML `cross-silo horizontal`** 路径上。
- 首选 backend：`MPI`。
- **禁止**把 `simulation + sp` 或 `simulation + MPI` 作为 Phase 1 完成标准。
- `cross-cloud` 可以作为备选宿主，但不是首个落地目标。

### 2.2 算法约束

以下逻辑必须完整保留：

- Phase 1: GA 零阶搜索
- Phase 2: 锚点归一化投影
- Phase 3: 服务器动量平滑
- 聚合后的 BN running stats 校准
- `trainable_mask` 对 BN buffers 的保护
- `global_model_buffer` / `velocity_buffer` 的跨轮状态保持

### 2.3 数据协议约束

必须保留 ShieldFL 的四分割语义：

- `server_val_set ∩ client_pool = ∅`
- `server_trust_set ∩ client_pool = ∅`
- `server_val_set ∩ server_trust_set = ∅`
- test 集封存，仅用于评估，不参与 GA fitness

### 2.4 设备与确定性约束

- 所有组件统一使用 `fedml.device.get_device(args)` 解析出的 `device`
- 禁止在新代码中写死 `cuda:0` / `torch.device("cuda")`
- CPU deterministic 模式下至少要显式设置：
  - 固定 `random_seed`
  - 固定 client sampling 顺序
  - 固定 aggregation 输入顺序
  - `torch.backends.cudnn.benchmark = False`
  - 必要时 `torch.use_deterministic_algorithms(True)`

---

## 3. Phase 1 允许缩小的只有“规模”，不是“逻辑”

为了适配纯 CPU 环境，Phase 1 可以降低实验规模；但**只能缩规模，不能减逻辑**。

### 3.1 允许缩小的内容

- `client_num_in_total`：例如从 10 降到 3 或 5
- `client_num_per_round`：例如全参与但总 client 更少
- `comm_round`：例如先做 3 轮或 5 轮
- 模型规模：可先用 `SimpleCNN` 做 CPU 可观测验证
- 数据规模：可缩小 `server_val_size` / `server_trust_size`
- batch size：可为 CPU 降到 16 / 32

### 3.2 不允许缩减的内容

- 不允许删除三阶段中的任何一阶段
- 不允许把 `aggregate()` 退回 FedAvg
- 不允许跳过 BN 校准（若模型含 BN）
- 不允许把输入顺序稳定性问题留到“后面再说”
- 不允许把 `server_val` 改成直接使用 test 集
- 不允许为 CPU 方便而改写 VeriFL 适应度定义
- 不允许为了先跑通而把 GA 默认流程改成 one-shot / greedy / 单客户端选择器

### 3.3 关于超参数缩小的判定

- `pop_size` / `generations` 的缩小 **只能用于 plumbing smoke test**。
- **只要目标是宣告 Phase 1 完成，就必须至少做一次使用 VeriFL-v16 默认 GA 预算的忠实运行**：
  - `pop_size = 15`
  - `generations = 10`

这条很重要：

> CPU 慢可以接受，算法被悄悄瘦身不接受。

---

## 4. Phase 1 的最小交付物

建议在 `python/examples/federate/prebuilt_jobs/shieldfl/` 下完成以下最小骨架：

- `main_fedml_shieldfl.py`
- `trainer/verifl_trainer.py`
- `trainer/verifl_aggregator.py`
- `trainer/micro_ga_base.py`
- `trainer/gpu_accelerator.py`
- `data/data_loader.py`
- `model/model_hub.py`
- `config/fedml_config_cpu_observable.yaml`
- `config/fedml_config_cpu_fidelity.yaml`
- `config/gpu_mapping.yaml`
- `utils/runtime.py`
- `README.md`（可后补）

---

## 5. Phase 1 实施顺序（压缩版）

下面这 12 步是建议的执行顺序；每一步都应有“可见输出”或“可验证状态”。

### Step 1：固定金标准来源

把以下文件明确标记为只读比对基线：

- `ShieldFL/src/strategies/ours/v16.py`
- `ShieldFL/src/strategies/ours/ga_base.py`
- `ShieldFL/src/strategies/ours/gpu_accelerator.py`
- `ShieldFL/src/core/evaluator.py`
- `ShieldFL/src/factories/data_factory.py`
- `ShieldFL/src/attacks/*`

### Step 2：先搭 FedML 宿主骨架

创建 `prebuilt_jobs/shieldfl/` 的入口与目录，但此时先不追求训练成功，只确认：

- `fedml.init()` 可读取自定义 YAML
- `fedml.device.get_device(args)` 可在 CPU 下解析为 `cpu`
- `FedMLRunner(args, device, dataset, model, trainer, aggregator)` 能正确构造

### Step 3：先把数据协议迁进去

在 `data/data_loader.py` 中先做的不是“能下载数据”，而是：

- 正确输出 FedML 标准 8-slot dataset tuple
- 同时生成 ShieldFL 的 `val/trust/eval bundle`
- 对互斥切分做硬断言，不要静默吞掉错误

### Step 4：迁模型入口

`model/model_hub.py` 先支持：

- `SimpleCNN`
- `ResNet20`

并确保：

- state_dict 键顺序稳定
- 与 ShieldFL 原模型权重结构兼容

### Step 5：迁客户端训练器

`VeriFLTrainer` 的第一版目标非常单纯：

- 能拿到本地数据
- 能执行 benign local SGD
- 能回传 `state_dict`
- `trainer.id` 与 `client_index` 对齐

此时先不接攻击也可以，但接口位要留好。

### Step 6：迁服务端聚合器主体

`VeriFLAggregator` 必须最先打通的是：

- `OrderedDict -> ndarray list`
- `ndarray list -> OrderedDict`
- stable sort client updates
- `aggregate()` 内完整调用 VeriFL-v16 三阶段逻辑

到这一步时，**哪怕精度还差，也必须保证代码路径是对的。**

### Step 7：迁 GPUAccelerator，但先保证 CPU fallback 完整

Phase 1 的 CPU 环境下，`GPUAccelerator` 的验收重点不是快，而是：

- CPU 模式逻辑与 GPU 模式同语义
- `trainable_mask` 正确
- BN 校准可在 CPU 路径工作
- 不隐式猜测设备

### Step 8：补 `runtime.py`

明确 runtime mode：

- `cpu-deterministic`
- `single-gpu-deterministic`（预留）
- `multi-gpu-throughput`（预留）

Phase 1 至少要让 `cpu-deterministic` 真正生效。

### Step 9：做 plumbing smoke test（不算完成，仅算通线）

允许用缩小配置快速验证：

- `SimpleCNN`
- 3 clients
- 1~3 rounds
- 可能临时缩小 `pop_size` / `generations`

但这一步只能回答：

> “FedML 宿主 + 自定义 trainer/aggregator 线通了没有？”

不能回答：

> “Phase 1 完成没有？”

### Step 10：做忠实 CPU observable run（Phase 1 必做）

这一步是核心验收：

- `cross-silo horizontal + MPI backend`
- `using_gpu: false`
- 保持 VeriFL-v16 默认核心超参数：
  - `pop_size = 15`
  - `generations = 10`
- 无攻击
- 至少 3 rounds
- 输出每轮：
  - loss
  - accuracy
  - 当前轮的关键聚合日志

### Step 11：做 CPU fidelity checkpoint（至少一次）

建议在 CPU 下再补一个更靠近论文主路径的忠实点：

- `ResNet20`
- CIFAR10
- 无攻击
- 1~3 rounds
- 确认 BN 路径和聚合状态路径都被真实触发

### Step 12：补最小对照结论

Phase 1 结束时必须写出一句清晰结论：

- **哪些结果已经证明“核心算法无逻辑降级地迁入了 FedML”**
- **哪些能力还只是预留，尚未验证**

---

## 6. 纯 CPU 环境的两档验证配置

为了兼顾“可观测结果”和“忠实迁移”，建议 Phase 1 直接准备两份 CPU 配置。

### 6.1 `cpu_observable`：先看到明显结果

用途：

- 快速确认自定义 trainer / aggregator / runtime / data 协议已通
- 在纯 CPU 上较快地产生可见 loss/accuracy 变化

建议参数：

- backend: `MPI`
- training_type: `cross_silo`
- `using_gpu: false`
- `client_num_in_total: 3`
- `client_num_per_round: 3`
- `comm_round: 3`
- `model: SimpleCNN`
- `batch_size: 16` 或 `32`
- `server_val_size: 100~200`
- `server_trust_size: 100~200`
- `attack_type: none`
- `pop_size: 15`
- `generations: 10`

### 6.2 `cpu_fidelity`：靠近主路径的忠实点

用途：

- 验证 ResNet20 + BN + VeriFL-v16 默认主路径
- 为后续 single-GPU 对齐打基础

建议参数：

- backend: `MPI`
- training_type: `cross_silo`
- `using_gpu: false`
- `client_num_in_total: 3`
- `client_num_per_round: 3`
- `comm_round: 1~3`
- `model: ResNet20`
- `batch_size: 16`
- `server_val_size: 100`
- `server_trust_size: 100`
- `attack_type: none`
- `pop_size: 15`
- `generations: 10`

---

## 7. Phase 1 的前置环境要求

如果目标是“在 FedML 框架中忠实验证自定义聚合器”，纯 CPU 环境至少需要满足：

### 必需项

- Python 环境可正常导入 `fedml`
- 可运行 `cross_silo` 路径
- 已安装 `mpi4py`
- 系统可用 `mpirun` / `mpiexec`

### 现实判定

如果当前纯 CPU 环境 **没有 MPI 运行时**，那么：

- 可以继续写代码
- 可以做本地单进程 harness 调试
- **但不能宣告 Phase 1 完成**

因为那样会退化成“代码看起来能跑”，而不是“FedML 宿主已忠实承载 VeriFL 聚合”。

---

## 8. Phase 1 明确不做的事

以下内容不属于 Phase 1：

- 全攻击菜单迁移完成
- Krum / RFA / Foolsgold 对照实验
- cross-cloud 部署打磨
- Launch / MLOps 全接通
- 多 GPU 吞吐优化
- DDP / DataParallel 单 client 多卡训练
- secure aggregation / DP 兼容性验证

这些都重要，但不是 Phase 1 的判题点。

---

## 9. Phase 1 完成标准

只有当下面 8 条全部满足，Phase 1 才算完成。

### 9.1 框架宿主完成

- [ ] `main_fedml_shieldfl.py` 运行在 FedML `cross-silo horizontal` 路径
- [ ] backend 为 `MPI`
- [ ] 自定义 `VeriFLTrainer` 和 `VeriFLAggregator` 被真实调用
- [ ] 没有退回内置 FedAvg 聚合

### 9.2 算法路径完成

- [ ] `aggregate()` 中真实执行 GA 搜索
- [ ] 真实执行 anchor projection
- [ ] 真实执行 server momentum
- [ ] ResNet20 路径下 BN recalibration 被触发
- [ ] BN buffers 未参与 anchor scaling

### 9.3 数据协议完成

- [ ] `server_val` / `server_trust` / `client_pool` 三者互斥
- [ ] test 集仅用于评估
- [ ] `val/trust` 资产没有丢失在 FedML dataset tuple 之外

### 9.4 CPU 可观测验证完成

- [ ] 在纯 CPU 环境下完成至少一次 `cpu_observable` 运行
- [ ] 至少 3 轮有可见 loss / accuracy 输出
- [ ] 聚合日志能证明三阶段都被执行
- [ ] 运行结束后能给出“明显可观测结果”，哪怕精度不高

### 9.5 忠实性验证完成

- [ ] 至少一次运行保留 VeriFL-v16 默认 `pop_size=15, generations=10`
- [ ] 没有用“缩小 GA 预算”的运行冒充最终验收
- [ ] 至少一次 `ResNet20` CPU fidelity checkpoint 成功完成

---

## 10. Phase 1 最终产出应该是什么

Phase 1 结束时，仓库里至少应该新增或定稿：

- `PHASE1.md`（本文）
- `python/examples/federate/prebuilt_jobs/shieldfl/main_fedml_shieldfl.py`
- `python/examples/federate/prebuilt_jobs/shieldfl/trainer/verifl_trainer.py`
- `python/examples/federate/prebuilt_jobs/shieldfl/trainer/verifl_aggregator.py`
- `python/examples/federate/prebuilt_jobs/shieldfl/trainer/micro_ga_base.py`
- `python/examples/federate/prebuilt_jobs/shieldfl/trainer/gpu_accelerator.py`
- `python/examples/federate/prebuilt_jobs/shieldfl/data/data_loader.py`
- `python/examples/federate/prebuilt_jobs/shieldfl/model/model_hub.py`
- `python/examples/federate/prebuilt_jobs/shieldfl/config/fedml_config_cpu_observable.yaml`
- `python/examples/federate/prebuilt_jobs/shieldfl/config/fedml_config_cpu_fidelity.yaml`
- 一份简短运行记录（哪怕只是 Markdown）说明：
  - 跑的是什么配置
  - 是否纯 CPU
  - 是否真的走了三阶段聚合
  - 输出结果是否可观测

---

## 11. 一句话版 Phase 1

> **先别急着把 ShieldFL 做成一个“功能很多”的 FedML 示例，而是先把 VeriFL-v16 的完整算法路径，原封不动地塞进 FedML `cross-silo horizontal` 宿主里，并在纯 CPU 上跑出能看得见的结果。**
