# Phase 0 详细实现规划（基于代码全量核验 + 勘误修正）

## 0. 代码实况基线

以下事实均由代码直接核验，非文档推断：

| 事实 | 代码证据 |
|------|---------|
| `main_fedml_shieldfl.py` 双分支选择 aggregator | L25-30：`aggregator_type` 判断 → `BaselineAggregator` 或 `VeriFLAggregator` |
| `VeriFLAggregator` 覆写 `on_before_aggregation` + `aggregate` | verifl_aggregator.py L75 + L141 |
| `BaselineAggregator` 也覆写 `aggregate`（E-4）| baseline_aggregator.py L38：含 Bulyan 分支，非 Bulyan 委托 `super()` |
| 两个 aggregator 都做 `setattr(args, "aggregator_type", ...)`（E-8）| verifl_aggregator.py L56 + baseline_aggregator.py L29 |
| `MetricsCollector.__init__` 读取 `aggregator_type` 默认值为 `"verifl"`（E-7）| metrics.py L30：`getattr(args, "aggregator_type", "verifl")` |
| FedML 基类 `ServerAggregator.aggregate()` 路径 | L85-92：`is_defense_enabled()` → True 走 `defend_on_aggregation()`；False 走 `FedMLAggOperator.agg()` |
| `FedMLDefender.defend_on_aggregation()` 内部再检查 `is_defense_on_aggregation()` | fedml_defender.py L172-178：不在白名单 → 回退 `base_aggregation_func()` 即 FedAvg |
| `enable_defense=false` 时 `FedMLDefender.init()` 设 `self.is_enabled=False` | fedml_defender.py L96：直接到 else 分支 |
| F-6 投毒旁路：`FedMLTrainer.train()` 传 `self.train_local` | fedml_trainer.py L76：`self.trainer.train(self.train_local, ...)` |
| `ClientTrainer.update_dataset()` 存毒数据到 `self.local_train_dataset` | client_trainer.py L48-58 |
| verifl_trainer.py `train()` 的参数 `train_data` 来源于 `self.train_local`（干净数据），从未读取 `self.local_train_dataset` | L53：`def train(self, train_data, device, args)` — 直接使用 `train_data` 参数 |
| run_experiment.sh 中 `aggregator_type` 从 `--aggregator` 参数获取 | run_experiment.sh L20：`AGGREGATOR="fedavg"` + L59：`--aggregator` handler + L291：`aggregator_type: "${AGGREGATOR}"` |
| `run_experiment.sh` 生成 `lambda_reg: 0.01`（非文档声称的 0.1）（E-3）| L251 |
| run_experiment.sh 生成 `server_momentum: 0.0`（E-2）| L248 |
| run_experiment.sh 生成 `max_samples_per_client: ${MAX_SAMPLES}`，默认 300（E-1/E-9）| L30 + L240 |
| `FedMLAggOperator.agg()` 使用样本数加权（非等权）| `agg_operator.py` L43：`w = local_sample_number / training_num` |
| `BaselineAggregator.__init__` 签名 = `(model, args, data_assets=None, device=None)`（E-5）| L20-31 |
| `VeriFLAggregator.__init__` 签名 = `(model, args, data_assets, device)` | L35 |
| 两个 aggregator 的 `test()` 逻辑高度相似（模型评估 + ASR + MetricsCollector）| baseline_aggregator.py L83-145，verifl_aggregator.py L290-340 |
| verifl_trainer.py 的 `train()` 中无标签分布日志（E-6）| L53-188：仅记录 loss |

---

## 1. P0.1：创建 ShieldFLAggregator

### 1.1 文件路径

`python/examples/federate/prebuilt_jobs/shieldfl/trainer/shieldfl_aggregator.py`

### 1.2 构造函数签名（E-5 修正）

```python
class ShieldFLAggregator(ServerAggregator):
    def __init__(self, model, args, data_assets=None, device=None):
        super().__init__(model, args)
        self.data_assets = data_assets
        self.device = device if device is not None else torch.device("cpu")
        self._last_agg_time = None

        metrics_dir = str(getattr(args, "metrics_output_dir", "./results"))
        self._metrics_collector = MetricsCollector(metrics_dir, args)
```

**为何需要 `data_assets` 和 `device`**（E-5 核验）：
- `test()` 方法中 ASR 评估调用 `evaluate_asr(model, test_loader=self.data_assets.test_loader, device, ...)`
- `test()` 方法中 `model.to(device)` 需要 `device`
- `MetricsCollector` 本身不依赖以上两者，但其初始化需在构造中完成

**为何不做 `setattr(args, "aggregator_type", ...)`**：
- 旧代码 `setattr` 的目的是强制覆写 `args` 供 `MetricsCollector` 读取
- P0.4 会将 run_experiment.sh 中 `aggregator_type` 改为固定值 `"shieldfl"`，**由 YAML 源头控制而非运行时覆写**
- 这消除了 `setattr` 的副作用风险（例如 FedML 框架其他组件通过 `args.aggregator_type` 做路由判断时被意外影响）

**`_last_agg_time` 的来源**：`BaselineAggregator` 在 `aggregate()` 中记录聚合耗时。由于 `ShieldFLAggregator` 不覆写 `aggregate()`，此值不会被赋值。保留字段供 `test()` → `MetricsCollector.log_round(agg_time=...)` 使用，值为 None 是安全的（`MetricsCollector.log_round` 已处理 `agg_time=None`）。

### 1.3 `get_model_params()` / `set_model_params()`

从两个旧 aggregator 中提取（一致且简单）：

```python
def get_model_params(self):
    return self.model.cpu().state_dict()

def set_model_params(self, model_parameters):
    self.model.load_state_dict(model_parameters, strict=True)
```

**基类核验**：`ServerAggregator.aggregate()` 在调用 `FedMLDefender.defend_on_aggregation()` 时传递 `extra_auxiliary_info=self.get_model_params()`。此处需要 `state_dict` 格式（OrderedDict），上述实现符合。

### 1.4 `test()` 方法

**从 `BaselineAggregator.test()` 迁移**（L83-142），因为：
- `BaselineAggregator.test()` 不引用 `self.gpu_accelerator`
- `VeriFLAggregator.test()` 额外做 `self.gpu_accelerator.device = device`——这是 VeriFL GA 评估的需求，与 ShieldFLAggregator（不含 GA）无关

迁移范围：
1. 模型评估循环（accuracy + loss）→ 原样复制
2. ASR 评估条件分支（`eval_asr` + `data_assets`）→ 原样复制
3. `MetricsCollector.log_round()` 调用 → 原样复制
4. 日志行将 `"BaselineAggregator test"` 改为 `"ShieldFLAggregator test"`

```python
def test(self, test_data, device, args):
    self.device = device
    model = self.model
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    metrics = {"test_correct": 0, "test_loss": 0.0, "test_total": 0, "test_accuracy": 0.0}
    with torch.no_grad():
        for images, labels in test_data:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            _, predicted = torch.max(logits, 1)
            metrics["test_correct"] += predicted.eq(labels).sum().item()
            metrics["test_loss"] += loss.item() * labels.size(0)
            metrics["test_total"] += labels.size(0)
    if metrics["test_total"] > 0:
        metrics["test_accuracy"] = metrics["test_correct"] / metrics["test_total"]
        metrics["test_loss"] /= metrics["test_total"]
    logging.info("ShieldFLAggregator test | loss=%.6f | accuracy=%.4f | samples=%s",
                 metrics["test_loss"], metrics["test_accuracy"], metrics["test_total"])

    asr_value = None
    if bool(getattr(args, "eval_asr", False)) and self.data_assets is not None:
        asr_result = evaluate_asr(
            model=model,
            test_loader=self.data_assets.test_loader,
            device=device,
            target_label=int(getattr(args, "target_label", 0)),
            trigger_size=int(getattr(args, "trigger_size", 3)),
            trigger_value=float(getattr(args, "trigger_value", 1.0)),
        )
        asr_value = asr_result["asr"]
        metrics.update({f"asr_{k}": v for k, v in asr_result.items()})

    if self._metrics_collector is not None:
        round_idx = int(getattr(args, "round_idx", -1))
        self._metrics_collector.log_round(
            round_idx=round_idx,
            test_accuracy=metrics["test_accuracy"],
            test_loss=metrics["test_loss"],
            test_total=int(metrics["test_total"]),
            asr=asr_value,
            agg_time=self._last_agg_time,
        )
    return metrics
```

### 1.5 `test_all()` 方法

```python
def test_all(self, train_data_local_dict, test_data_local_dict, device, args) -> bool:
    return False
```

与两个旧 aggregator 行为一致。

### 1.6 所需 imports

```python
import logging
from typing import Dict, Optional
import torch
import torch.nn as nn
from fedml.core.alg_frame.server_aggregator import ServerAggregator
from eval.asr import evaluate_asr
from eval.metrics import MetricsCollector
```

**核验**：`ServerAggregator` 的 import 路径与 verifl_aggregator.py 一致。`evaluate_asr` 和 `MetricsCollector` 的 import 路径与 baseline_aggregator.py 一致。

### 1.7 禁止列表（不可出现的内容）

| 禁止项 | 理由 |
|--------|------|
| `def on_before_aggregation` | 基类已完整处理攻击/防御调度 |
| `def aggregate` | 基类已根据 `is_defense_enabled()` 自动切换 FedAvg / defense |
| `def on_after_aggregation` | 基类已处理 DP noise + defense after agg |
| `import GPUAccelerator` | VeriFL 专属组件 |
| `import MicroGABase` | VeriFL 专属组件 |
| `setattr(args, "aggregator_type", ...)` | YAML 源头控制（P0.4） |
| `self.server_momentum` | 不在 Phase 0 |
| `self.velocity_buffer` | 不在 Phase 0 |
| `self.global_model_buffer` | 不在 Phase 0 |

### 1.8 基类管线路径核验

当 `ShieldFLAggregator` 不覆写任何管线方法时，实际执行路径为：

**`defense_type=none`, `enable_defense=false`（Phase 0/1）：**

```
ServerAggregator.on_before_aggregation()
├─ FedMLDifferentialPrivacy.is_global_dp_enabled()     → False（未配置 DP）
├─ FedMLAttacker.is_data_reconstruction_attack()       → False
├─ FedMLAttacker.is_model_attack()                     → True（仅 model_replacement 实验时）
│  └─ attack_model() → 对恶意客户端应用 γ 缩放
├─ FedMLDefender.is_defense_enabled()                  → False
└─ return (processed_list, client_idxs)

ServerAggregator.aggregate()
├─ FedMLDefender.is_defense_enabled()                  → False
└─ FedMLAggOperator.agg(self.args, raw_list)           → 样本数加权 FedAvg

ServerAggregator.on_after_aggregation()
├─ FedMLDifferentialPrivacy.is_global_dp_enabled()     → False
├─ FedMLDefender.is_defense_enabled()                  → False
└─ return aggregated_model（原样返回）
```

**关键确认**：`FedMLAttacker.is_model_attack()` 在 `on_before_aggregation` 中被基类正确处理。`VeriFLAggregator` 的 `on_before_aggregation` 自定义实现与基类等价（都调用 `FedMLAttacker.attack_model()`），因此 Scaling Attack 的 γ 缩放在新路径下不受影响。

---

## 2. P0.2：简化 main_fedml_shieldfl.py

### 2.1 当前代码（需替换的部分）

```python
from trainer.verifl_aggregator import VeriFLAggregator
from trainer.baseline_aggregator import BaselineAggregator
from trainer.verifl_trainer import VeriFLTrainer
```
```python
aggregator_type = str(getattr(args, "aggregator_type", "verifl")).strip().lower()
if aggregator_type == "fedavg":
    aggregator = BaselineAggregator(model=model, args=args, data_assets=data_assets, device=device)
else:
    aggregator = VeriFLAggregator(model=model, args=args, data_assets=data_assets, device=device)
```

### 2.2 修改后

```python
from trainer.shieldfl_aggregator import ShieldFLAggregator
from trainer.verifl_trainer import VeriFLTrainer
```
```python
aggregator = ShieldFLAggregator(model=model, args=args, data_assets=data_assets, device=device)
```

### 2.3 移除项

- `from trainer.verifl_aggregator import VeriFLAggregator` ← 删除
- `from trainer.baseline_aggregator import BaselineAggregator` ← 删除
- `aggregator_type = str(getattr(...))` 判断块 ← 删除

### 2.4 保留项（不变）

- `from trainer.verifl_trainer import VeriFLTrainer` ← 保留（trainer 与 aggregator 独立）
- `trainer = VeriFLTrainer(model=model, args=args)` ← 保留
- `dataset, data_assets = load_shieldfl_data(args)` ← 保留
- `fedml_runner = FedMLRunner(args, device, dataset, model, trainer, aggregator)` ← 保留

### 2.5 `FedMLRunner` 构造器验证

`FedMLRunner` 接受 `(args, device, dataset, model, trainer, aggregator)` 参数。`aggregator` 参数的类型要求是 `ServerAggregator` 子类。`ShieldFLAggregator` 继承 `ServerAggregator`，类型匹配。

---

## 3. P0.3：修复 F-6（投毒数据旁路）

### 3.1 修改文件

verifl_trainer.py

### 3.2 修改位置

`train()` 方法开头（L53 之后，现有逻辑之前）

### 3.3 添加代码

```python
def train(self, train_data, device, args):
    # ------------------------------------------------------------------
    # F-6 fix: FedMLTrainer.train() always passes self.train_local (the
    # *original* clean DataLoader).  ClientTrainer.update_dataset() stores
    # the poisoned DataLoader in self.local_train_dataset, but train()
    # never reads it.  Redirect here so data-poisoning attacks (Label
    # Flipping) actually affect training.
    #
    # Safety note (E-10): after update_dataset(), local_train_dataset is
    # non-None for ALL clients (including benign ones).  For benign
    # clients, local_train_dataset == the clean DataLoader passed in,
    # so the redirect is a no-op (same object reference).
    # ------------------------------------------------------------------
    if hasattr(self, 'local_train_dataset') and self.local_train_dataset is not None:
        train_data = self.local_train_dataset

    # --- F-6 diagnostic: log label distribution (E-6) ---
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        _label_counts = {}
        for _, _labels in train_data:
            for _l in _labels.tolist():
                _label_counts[_l] = _label_counts.get(_l, 0) + 1
        logging.debug(
            "Client %s train label_distribution=%s",
            getattr(self, 'id', '?'),
            dict(sorted(_label_counts.items()))
        )

    # --- Scaling attack: determine if this client poisons this round ---
    # ... (existing code unchanged from here)
```

### 3.4 E-10 安全性分析（非恶意客户端的路径）

代码流程追踪：

1. `ClientTrainer.update_dataset()` L48-58：
   - **LF 恶意客户端**: `is_data_poisoning_attack()=True`, `is_to_poison_data(client_id)=True` → `self.local_train_dataset = poison_data(local_train_dataset)` → **新 DataLoader 对象**
   - **LF 非恶意客户端**: `is_data_poisoning_attack()=True`, `is_to_poison_data(client_id)=False` → `self.local_train_dataset = local_train_dataset` → **同一引用**
   - **Scaling 所有客户端**: `is_data_poisoning_attack()=False` → `self.local_train_dataset = local_train_dataset` → **同一引用**
   - **无攻击**: 同 Scaling

2. `FedMLTrainer.train()` L76：`self.trainer.train(self.train_local, ...)` 传入 `self.train_local`

3. F-6 修复逻辑：`train_data = self.local_train_dataset`
   - 对 LF 恶意客户端：替换为投毒数据 ✅
   - 对其他所有客户端：`self.local_train_dataset` 与 `self.train_local` 指向同一对象 → **无变化**（等价于不做替换）✅

4. `self.train_local` 的赋值位置：`FedMLTrainer.update_dataset()` L54：`self.train_local = self.train_data_local_dict[client_index]`

5. `self.trainer.update_dataset()` 的调用位置：`FedMLTrainer.update_dataset()` L69：`self.trainer.update_dataset(self.train_local, self.test_local, self.local_sample_number)` — 注意传入的是**同一个** `self.train_local`

**结论**：对非恶意客户端，`self.local_train_dataset is self.train_local` 为 True（同一对象引用）。F-6 修复的赋值 `train_data = self.local_train_dataset` 在功能上等价于 `train_data = train_data`（no-op）。**安全**。

### 3.5 E-6 诊断日志选择

**方案 A**：遍历 DataLoader 统计标签分布（文档方案）
- 问题：遍历整个 DataLoader 有计算开销；DataLoader 是迭代器，遍历后需重置
- 修正：使用 `logging.DEBUG` 级别，且仅在调试时遍历。但注意 DataLoader 遍历后 epoch 序被消耗。

**方案 B**（采纳）：使用 `logging.DEBUG` 级别 + 对 DataLoader 的 dataset 属性统计，避免消耗迭代器
- 但不是所有 DataLoader 都暴露 `.dataset.targets`/`.dataset.tensors`，取决于 `data_loader.py` 的实现

**最终决策**：使用方案 A 但置于 `logging.DEBUG` 级别，因为：
1. 正常运行不触发（INFO 不触发 DEBUG）
2. 验证 AC-P0-6 时可临时开启 `--log_level DEBUG`
3. DataLoader 的消费仅发生一次（训练循环本身也要遍历，放在循环前可先采集分布）

**但存在实际问题**：如果 `train_data` 是原生 `DataLoader`（非可重置的），遍历一次后训练循环将得到空迭代器。

**安全修正**：改为统计 DataLoader 底层 dataset 的标签：

```python
if logging.getLogger().isEnabledFor(logging.DEBUG):
    _ds = getattr(train_data, 'dataset', None)
    if _ds is not None:
        _targets = getattr(_ds, 'targets', getattr(_ds, 'tensors', [None])[-1])
        if _targets is not None:
            import collections
            _dist = dict(collections.Counter(
                int(t) for t in (_targets if not callable(getattr(_targets, 'tolist', None)) else _targets.tolist())
            ))
            logging.debug("Client %s label_dist=%s", getattr(self, 'id', '?'), _dist)
```

**问题**：这依赖于 `data_loader.py` 的具体 Dataset 实现。如果使用 TensorDataset，标签在 `tensors[-1]`；如果使用包装类，可能在 `targets` 属性。

**最保守的方案（推荐）**：不在代码中添加标签分布日志。将 AC-P0-6 的验证方式改为：

> AC-P0-6（修订）：F-6 修复验证：运行 `attack_type=label_flipping` 的 5 轮 smoke test，对比恶意客户端首轮 train loss 与 clean baseline 首轮 train loss。Label Flipping 使标签 100% 错误，**恶意客户端首轮 loss 应显著高于 clean baseline**（接近 $-\ln(0.1) \approx 2.3$ 量级，因随机猜测 10 分类 loss ≈ 2.3）。若两者 loss 接近（差异 < 0.5），说明 F-6 修复未生效。

**为何此方案可靠**：
- verifl_trainer.py L170-175 已记录每个 epoch 的 train loss（`logging.info("epoch=%d loss=%.6f ...")`）
- Label Flipping 将 100% 标签翻转为错误标签，模型在首轮无法学到有效特征，loss 接近随机猜测
- 这无需任何新代码变更即可验证

---

## 4. P0.4：更新 run_experiment.sh

### 4.1 移除 `--aggregator` 参数

**删除位置**：
- L20：`AGGREGATOR="fedavg"` → 删除
- L59-61：`--aggregator) AGGREGATOR="$2"; shift 2 ;;` → 删除

### 4.2 修改 YAML 中 `aggregator_type`

**当前**（L291）：
```yaml
  aggregator_type: "${AGGREGATOR}"
```

**修改为**：
```yaml
  aggregator_type: "shieldfl"
```

**理由（E-7）**：固定值避免 `MetricsCollector` 在缺少 `aggregator_type` 时回退到默认值 `"verifl"`。选择 `"shieldfl"` 而非 `"fedavg"` 是因为实际防御策略由 `defense_type` 控制，`aggregator_type` 仅用于指标文件名标识——标识为 `"shieldfl"` 更准确地反映实际使用的 aggregator 类。

### 4.3 移除 echo 中的 aggregator 行

**当前**（L300）：
```bash
echo "  aggregator=${AGGREGATOR} pmr=${PMR} alpha=${ALPHA} seed=${SEED}"
```

**修改为**：
```bash
echo "  pmr=${PMR} alpha=${ALPHA} seed=${SEED}"
```

### 4.4 移除持久化配置名中的 aggregator 片段

**当前**（L309）：
```bash
PERSIST_NAME="config_${MODEL}_${DATASET}_${AGGREGATOR}_atk${ATTACK}_def${DEFENSE}_a${ALPHA}_pmr${PMR}_seed${SEED}.yaml"
```

**修改为**：
```bash
PERSIST_NAME="config_${MODEL}_${DATASET}_shieldfl_atk${ATTACK}_def${DEFENSE}_a${ALPHA}_pmr${PMR}_seed${SEED}.yaml"
```

### 4.5 GPU 模式自动升级 `RUNTIME_MODE`（E-13 计划外修复）

**问题**：`RUNTIME_MODE` 默认为 `"cpu-deterministic"`。当用户指定 `--gpu` 时，`runtime.py` 中 `torch.use_deterministic_algorithms(True)` 在 GPU 上执行，但 `CUBLAS_WORKSPACE_CONFIG` 仅在 `"single-gpu-deterministic"` 模式下设置（`runtime.py` L43）。导致 GPU smoke test 崩溃：`RuntimeError: CUBLAS_WORKSPACE_CONFIG not set`。

**修复位置**：`run_experiment.sh` GPU 模式块（与 `CPU_TRANSFER="true"` 同处）：

```bash
if [[ "$GPU" == "true" ]]; then
	CPU_TRANSFER="true"
	# runtime_mode 未被用户显式覆写时，自动升级为 GPU 确定性模式
	if [[ "$RUNTIME_MODE" == "cpu-deterministic" ]]; then
		RUNTIME_MODE="single-gpu-deterministic"
	fi
fi
```

**安全性**：仅在用户未通过 `--runtime` 显式指定模式时生效。若用户已指定 `single-gpu-fast` 等非默认模式，不会被覆盖。此修改在 YAML 配置生成之前执行，确保 `shieldfl_args.runtime_mode` 写入正确值。

### 4.6 `defense_type` / `enable_defense` 逻辑（保持不变）

当前逻辑已经正确：
```bash
ENABLE_DEFENSE="false"
DEFENSE_TYPE="none"
if [[ "$DEFENSE" != "none" ]]; then
    ENABLE_DEFENSE="true"
    DEFENSE_TYPE="$DEFENSE"
fi
```

`--defense none` → `enable_defense: false`, `defense_type: "none"` → `FedMLDefender.is_enabled=False` → FedAvg ✅

### 4.7 `byzantine_client_num` 计算（保持不变，P1 阶段通过 `--pmr` 参数控制）

当前：`BYZANTINE_NUM=$(python3 -c "import math; print(max(1, math.ceil($CLIENTS * $PMR)))")`

Phase 0 测试中 `--attack none` → `BYZANTINE_NUM=0`，不受影响。
Phase 1 中 `--pmr 0.1 --clients 10` → `ceil(10×0.1)=1` → K=1 ✅

### 4.8 `sort_client_updates` 参数（保持不变）

当前在 `shieldfl_args` 中：`sort_client_updates: true`。此参数在 `VeriFLAggregator` 中使用（排序客户端更新以确保确定性）。`ShieldFLAggregator` 不使用它。它留在 YAML 中不会产生副作用（`getattr` 读取不到就被忽略）。Phase 2 迁移 VeriFL 时可能需要。暂保留。

---

## 5. P0.5：基线回归验证（门禁）- 修订版

### 5.1 验证条件

| 编号 | 条件 | 验证方式 | 状态 |
|------|------|---------|------|
| AC-P0-1 | exit code = 0，生成 metrics JSONL 文件 | `echo $?` + `ls results/metrics_*.jsonl` | 可直接验证 |
| AC-P0-2 | 日志中 `FedMLAggOperator.agg()` 被调用 | `grep "FedMLAggOperator" log/` — **注意：需核验 `agg_operator.py` 是否有日志输出** | 见下方分析 |
| AC-P0-3 | 日志中不出现 GA/L2 投影/momentum/BN 重校准字样 | `grep -i "GA search\|L2 projection\|momentum\|recalibrat" log/` | 可直接验证 |
| AC-P0-4 | `shieldfl_aggregator.py` 中不存在 `def on_before_aggregation` / `def aggregate` / `def on_after_aggregation` | `grep "def on_before_aggregation\|def aggregate\|def on_after_aggregation" shieldfl_aggregator.py` | 可直接验证 |
| AC-P0-5 | 全项目中 `VeriFLAggregator` / `BaselineAggregator` 无活跃引用（import / 实例化） | `grep -r "VeriFLAggregator\|BaselineAggregator" python/ --include="*.py" \| grep -v ".deprecated\|__pycache__\|#"` | 可直接验证 |
| AC-P0-6 | **（修订）** F-6 修复验证：`attack_type=label_flipping` 5 轮 smoke test 中，恶意客户端首轮 train loss ≥ 2.0（Label Flipping 导致 100% 错误标签，loss 趋近随机猜测 $\approx 2.3$），且 clean baseline 同配置首轮 loss < 1.5 | 对比两次运行日志中 `epoch=0 loss=` 值 | 可直接验证 |

### 5.2 AC-P0-2 核验补充

检查 `agg_operator.py` 是否有日志输出：

```python
# agg_operator.py 中 FedAvg 分支：
if args.federated_optimizer == "FedAvg":
    ...
```

**实际情况**：`agg_operator.py` 的 FedAvg 分支没有 `logging.info` 调用。无法通过 grep 日志验证 `FedMLAggOperator.agg()` 被调用。

**替代验证方式**：
- **方式 A**：在 smoke test 前临时给 `agg_operator.py` 加一行日志（但违反冻结规则）
- **方式 B（推荐）**：通过**排除法**验证——AC-P0-3 确认日志中不出现任何 VeriFL 特有字样（GA/L2/momentum/BN），加上 AC-P0-4 确认代码不覆写管线，等价于证明聚合路径必经 `FedMLAggOperator.agg()`

**修订 AC-P0-2**：

> AC-P0-2（修订）：日志中不出现 `"VeriFL"` / `"GA "` / `"L2 "` / `"momentum"` / `"recalibrat"` / `"Phase 1"` / `"Phase 2"` / `"Phase 3"` / `"Phase 4"` 关键词（`VeriFLAggregator.aggregate()` 的特征日志）。结合 AC-P0-4，排除法证明聚合路径走基类 → `FedMLAggOperator.agg()`。

### 5.3 Smoke Test 运行命令

```bash
# Clean baseline smoke test
bash scripts/run_experiment.sh \
  --model ResNet18 --dataset cifar10 \
  --attack none --defense none \
  --alpha 0.5 --seed 0 \
  --rounds 5 --clients 10 --epochs 1 --batch_size 64

# F-6 验证 smoke test
bash scripts/run_experiment.sh \
  --model ResNet18 --dataset cifar10 \
  --attack label_flipping --defense none \
  --pmr 0.3 --alpha 0.5 --seed 0 \
  --rounds 5 --clients 10 --epochs 1 --batch_size 64
```

---

## 6. Phase 0 文件变更总表

| 操作 | 文件 | 变更类型 | 变更量 |
|------|------|---------|-------|
| **新建** | `trainer/shieldfl_aggregator.py` | P0.1 | ~90 行 |
| **修改** | `main_fedml_shieldfl.py` | P0.2 | 删 2 import + 删 4 行分支 + 改 1 行实例化 |
| **修改** | `trainer/__init__.py` | P0.2 补遗 | re-export 从 `VeriFLAggregator` 改为 `ShieldFLAggregator`（见 E-12） |
| **修改** | `trainer/verifl_trainer.py` | P0.3 | 添加 ~8 行（F-6 修复 + 注释）|
| **修改** | `scripts/run_experiment.sh` | P0.4 | 删 `--aggregator` 处理 + 改 `aggregator_type` 为固定值 + GPU 模式自动升级 `RUNTIME_MODE`（E-13） |
| **不变** | `trainer/verifl_aggregator.py` | — | 保留不动（Phase 2 废弃）|
| **不变** | `trainer/baseline_aggregator.py` | — | 保留不动（Phase 2 废弃）|
| **不变** | 所有 §6.2 冻结文件 | — | 不触碰 |

---

## 7. 冻结参数补充（E-9）

以下参数在 Phase 0 期间必须保持不变（任何改动可能破坏等价性验证和后续实验基准）：

| 参数 | 值 | 来源 | 影响 |
|------|-----|------|------|
| `max_samples_per_client` | 300 | run_experiment.sh L30 | E-1：使样本加权 FedAvg 退化为等权 FedAvg |
| `val_per_class` | 50 | run_experiment.sh L31 | VeriFL fitness 评估质量 |
| `test_subset_size` | 500 | run_experiment.sh L32 | ACC/ASR 评估方差 |
| `server_momentum` | 0.0 | run_experiment.sh L248 | E-2：VeriFL Phase 3 是否生效 |
| `server_lr` | 1.0 | run_experiment.sh L36 | 同上 |
| `lambda_reg` | 0.01 | run_experiment.sh L251 | E-3：VeriFL 正则项系数 |
| `federated_optimizer` | `"FedAvg"` | run_experiment.sh L235 | 聚合算子选择 |
| 全部 §6.1 参数 | （见文档）| — | — |

---

## 8. 风险-应对映射（Phase 0 特有）

| 风险 | 触发条件 | 诊断方法 | 应对 |
|------|---------|---------|------|
| R-P0-1：`FedMLRunner` 不接受新 aggregator | 类型检查失败 | 运行 smoke test 观察 import/type error | `ShieldFLAggregator` 继承 `ServerAggregator`，类型一致。若失败只可能是 import 路径问题 |
| R-P0-2：`test()` 方法签名不匹配框架调用 | 框架调用时参数数量不对 | 运行 smoke test 观察 TypeError | 签名 `test(self, test_data, device, args)` 与两个旧 aggregator 完全一致，框架调用方式相同 |
| R-P0-3：MetricsCollector 文件名意外变化 | 包含 `"verifl"` 而非 `"shieldfl"` | 检查 `results/metrics_*.jsonl` 文件名 | P0.4 固定 `aggregator_type: "shieldfl"` 应解决。若仍含 `"verifl"`，检查 `fedml.init()` 是否正确解析 `shieldfl_args` 节 |
| R-P0-4：F-6 修复后非恶意客户端行为变化 | 非恶意客户端 accuracy 与 M1.5 baseline 不一致 | 对比 clean baseline 指标 | E-10 分析显示 no-op；若出现差异，检查 `self.local_train_dataset` 与 `self.train_local` 是否确实同一引用 |
| R-P0-5：`shieldfl_args` 不被 `fedml.init()` 合并到 `args` | `args.aggregator_type` 读不到 | `print(dir(args))` 检查属性 | FedML 的 `fedml.init()` 将所有 YAML 顶层和嵌套键展平到 `args`。如果 `shieldfl_args` 不被自动展平，需手动在 main 中 `args.__dict__.update(args.shieldfl_args)` |

### R-P0-5 详细预判

**核验需求**：`fedml.init()` 是否将 `shieldfl_args.aggregator_type` 展平到 `args.aggregator_type`？

当前代码证据：`main_fedml_shieldfl.py` L25 `getattr(args, "aggregator_type", "verifl")` 在旧代码中能读到值（来自 YAML 的 `shieldfl_args.aggregator_type`），说明 FedML **确实展平了** `shieldfl_args`。或者，verifl_aggregator.py 的 `setattr(args, "aggregator_type", "verifl")` 覆写导致总能读到——无论 YAML 是否展平。

**验证方案**：在 P0.5 smoke test 中，检查 metrics 文件名是否包含 `shieldfl`。如果包含 `verifl`（MetricsCollector 的默认值），说明 `args.aggregator_type` 未从 YAML 读到，需在 `main_fedml_shieldfl.py` 中手动合并 `shieldfl_args`。

---

## 9. 实施顺序（步骤级）

```
Step 1: 创建 shieldfl_aggregator.py（P0.1）
Step 2: 修改 main_fedml_shieldfl.py（P0.2）
Step 2.5: 修改 trainer/__init__.py（P0.2 补遗 E-12）
Step 3: 修改 verifl_trainer.py（P0.3 F-6 修复）
Step 4: 修改 run_experiment.sh（P0.4）
Step 5: 代码静态检查
  ├─ grep shieldfl_aggregator.py 确认无禁止项（AC-P0-4）
  ├─ grep 活跃引用（AC-P0-5）
  └─ python -c "import ast; ast.parse(open('...').read())" 语法检查
Step 6: Clean baseline smoke test（AC-P0-1~P0-3）
Step 7: LF smoke test（AC-P0-6）
Step 8: 结果确认 → Phase 0 完成
```

---

## 10. 实施勘误（代码全量核验时发现的遗漏）

### E-12 🟠 `trainer/__init__.py` 遗漏更新

**问题**：§6 文件变更总表中未列出 `trainer/__init__.py`。该文件包含活跃 re-export：

```python
from .verifl_aggregator import VeriFLAggregator
__all__ = ["VeriFLTrainer", "VeriFLAggregator"]
```

这是一个**活跃 import 引用**（模块加载时执行），会导致 AC-P0-5 失败。

**决策依据**：
- M2_SA_失败导致的修复和验证.md §2 P0.2 明确要求 "移除所有 aggregator_type 分支逻辑和**相关 import**"
- AC-P0-5 要求 "全项目 grep 不出 `VeriFLAggregator` 或 `BaselineAggregator` 的活跃引用"
- `__init__.py` 的 re-export 属于活跃引用范畴，非定义文件内部引用

**处置决策**：将 `trainer/__init__.py` 的 re-export 从 `VeriFLAggregator` 改为 `ShieldFLAggregator`，保留 `VeriFLTrainer`（仍在使用）。

**修改后**：

```python
from .verifl_trainer import VeriFLTrainer
from .shieldfl_aggregator import ShieldFLAggregator

__all__ = ["VeriFLTrainer", "ShieldFLAggregator"]
```

**AC-P0-5 影响**：修复后，活跃引用仅剩旧 aggregator 定义文件内部（`verifl_aggregator.py`、`baseline_aggregator.py` 的 class 声明和日志字符串），这些文件在 Phase 2 中标记为废弃，不构成活跃 import/实例化引用。

### E-13 🟠 GPU 模式下 `RUNTIME_MODE` 未自动升级导致 `CUBLAS_WORKSPACE_CONFIG` 崩溃

**问题**：`run_experiment.sh` 的 `RUNTIME_MODE` 默认值为 `"cpu-deterministic"`。当用户指定 `--gpu` 但未显式指定 `--runtime single-gpu-deterministic` 时，`runtime.py` 的 `configure_runtime()` 在 `"cpu-deterministic"` 模式下调用 `torch.use_deterministic_algorithms(True)` 但**不设置** `CUBLAS_WORKSPACE_CONFIG`（该环境变量仅在 `"single-gpu-deterministic"` 分支中通过 `os.environ.setdefault` 设置）。在 GPU 上执行 `loss.backward()` 时 CUBLAS 操作触发 `RuntimeError`。

**根因**：`runtime.py` 的设计假设调用方在 GPU 场景下传入 `"single-gpu-deterministic"`，但 `run_experiment.sh` 未做此映射。

**修复**：在 `run_experiment.sh` 的 GPU 模式块中，当 `RUNTIME_MODE` 仍为默认值 `"cpu-deterministic"` 时自动升级为 `"single-gpu-deterministic"`。此修改在 YAML 配置生成之前执行，确保 `shieldfl_args.runtime_mode` 写入正确值。

**影响**：仅影响 GPU 运行场景。CPU 运行不受影响。用户通过 `--runtime` 显式指定模式时不会被覆盖。