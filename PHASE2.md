# ShieldFL → FedML Phase 2 实施清单

> 本文基于三份上游文档的交集推导：
> - **MOVE.md P1**：攻击迁入、ASR 评估、BN/momentum 一致性验证
> - **学术需求.md**：M1 基线可信 → M2 攻击生效 → M3 防御对照 → M4 结果固化
> - **PHASE1.md / RUN_RECORD_PHASE1.md**：Phase 1 已交付的基础设施
>
> Phase 2 的关键词：**在 Phase 1 已验证的 VeriFL 宿主上，补齐攻防实验全链路，使项目达到"可跑对照实验"的学术基础设施状态。**

---

## 1. Phase 2 的目标

Phase 2 必须同时满足两件事：

1. **工程目标（对应 MOVE P1）**：攻击可注入、ASR 可评估、聚合时间可度量、BN 校准与 server momentum 跨轮状态一致性可验证。
2. **学术目标（对应学术需求 M1–M3 基础设施）**：两条任务线（`ResNet18 + CIFAR10`、`LeNet5 + MNIST`）的数据/模型/攻击/防御/指标全链路就绪，可生产 `mean ± std` 对照结果。

Phase 2 结束时的状态应为：

> **在纯 CPU 上可运行涵盖 attack × defense × PMR × α × seed 组合的轻量 smoke test，并产出结构化逐轮指标文件，为后续 GPU 大规模实验做好基础设施准备。**

---

## 2. 与学术里程碑的对应关系

| 学术里程碑 | Phase 2 覆盖范围 | 说明 |
|---|---|---|
| **M1：基线可信** | ✅ 完整覆盖 | FedAvg baseline 在两条任务线上可跑，逐轮输出 MA/Loss |
| **M2：攻击生效** | ✅ 完整覆盖 | 三类攻击 × FedAvg × PMR/α/seed 组合可跑通 |
| **M3：防御对照** | ✅ 基础设施就绪 | FedML 内置防御可通过 YAML 切换；VeriFL 聚合器已接入攻击钩子 |
| **M4：结果固化** | ⬜ 仅准备指标采集层 | 自动汇总统计留到有 GPU 资源后的正式实验阶段 |

---

## 3. Phase 2 硬约束

### 3.1 攻防实现来源

学术需求明确：**攻防算法统一采用 FedML 内置实现**。

- 攻击走 `FedMLAttacker` 路径（`enable_attack: true` + `attack_type`）
- 防御走 `FedMLDefender` 路径（`enable_defense: true` + `defense_type`）
- ShieldFL 原有攻击类（`ShieldFL/src/attacks/*`）仅作为参考实现，不作为运行时主线
- 例外：`BulyanDefense` 存在于 FedML 代码库但未注册到 `FedMLDefender`，需做最小适配

### 3.2 数据协议升级

学术需求要求 Root Dataset 采用**分层均衡采样（stratified balanced）**：

- CIFAR-10：`server_val` / `server_trust` 默认每类等量（推荐每类 50）
- MNIST：同口径，各类等量
- 旧实现使用全局随机打乱后切片，仅满足集合隔离，不满足类别均衡 → **必须升级**

### 3.3 超参数公平性

- 本地训练预算（epochs, batch_size, client_fraction, comm_round）必须跨所有攻防组合保持一致
- 攻击/防御专属超参数必须显式声明
- 禁止双重标准调参

### 3.4 VeriFL 实验的双重防御隔离

- 当使用 VeriFL 聚合器时，必须确保 FedML 内置防御处于关闭状态，避免双重防御污染
- 当使用 FedML 内置防御时，必须使用 FedAvg 聚合器（不经过 VeriFL 三阶段）

### 3.5 设备约束

- 当前测试环境无 GPU
- 所有验证在 CPU 下进行
- `single-gpu-deterministic` 运行档位代码要写好，但 GPU 测试留到有资源时执行

---

## 4. 当前代码现状与差距分析

### 4.1 已就绪（Phase 1 交付物）

| 能力 | 文件 | 状态 |
|---|---|---|
| FedML cross-silo + MPI 宿主 | `main_fedml_shieldfl.py` | ✅ |
| VeriFL 三阶段聚合 | `trainer/verifl_aggregator.py` | ✅ |
| GA 搜索 + 锚点投影 + 服务器动量 | 同上 | ✅ |
| BN 校准 | `trainer/gpu_accelerator.py` | ✅ |
| CIFAR-10 数据加载（随机切片） | `data/data_loader.py` | ✅（需升级采样策略） |
| SimpleCNN / ResNet20 模型 | `model/` | ✅ |
| cpu-deterministic 运行模式 | `utils/runtime.py` | ✅ |
| 无攻击场景 smoke test | 配置YAML + 运行记录 | ✅ |

### 4.2 需新增

| 能力 | 差距 |
|---|---|
| MNIST 数据加载 | `data_loader.py` 仅支持 CIFAR-10，`dataset != CIFAR10` 直接报错 |
| ResNet18 模型 | model_hub 未注册，无实现文件 |
| LeNet-5 模型（MNIST） | 无实现 |
| 攻击注入 | `VeriFLTrainer.train()` 对 `attack_type != "none"` 直接抛 `NotImplementedError` |
| FedMLAttacker 钩子 | `VeriFLAggregator.on_before_aggregation()` 覆盖了基类，未调用 FedMLAttacker 模型攻击钩子 |
| FedAvg 基线聚合器 | 无独立 FedAvg 聚合器；M1/M2 实验不应使用 VeriFL 聚合逻辑 |
| ASR 评估 | `test()` 方法仅计算 MA/Loss，无后门 ASR 指标 |
| 聚合时间度量 | `aggregate()` 未记录计时 |
| 分层均衡采样 | 当前 val/trust 取自全局随机打乱切片 |
| 实验编排脚本 | 无一键运行 attack × defense × PMR × α × seed 的能力 |
| 结构化指标输出 | 无逐轮 JSON/CSV 指标落盘 |
| Bulyan 防御接入 | FedML 有实现文件但未注册到 FedMLDefender |
| TPR/FPR 指标 | 无客户端筛选结果追踪 |

---

## 5. Phase 2 实施步骤

### Step 1：升级数据加载器——新增 MNIST 支持 + 分层均衡采样

**文件**：`data/data_loader.py`

**改动概述**：

1. **新增 `_load_mnist()` 函数**：

```python
def _load_mnist(data_path: str):
    """加载 MNIST 数据集，返回 (trainset, testset, server_val_base_dataset)"""
    root_path = Path(data_path).expanduser().resolve()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    trainset = datasets.MNIST(root=str(root_path), train=True, download=True, transform=transform)
    testset = datasets.MNIST(root=str(root_path), train=False, download=True, transform=transform)
    server_val_base_dataset = datasets.MNIST(root=str(root_path), train=True, download=False, transform=transform)
    return trainset, testset, server_val_base_dataset
```

2. **新增 `_stratified_balanced_sample()` 函数**，替代旧的全局随机打乱切片逻辑：

```python
def _stratified_balanced_sample(targets, num_per_class, num_classes, seed):
    """
    分层均衡采样：每类取 num_per_class 个样本。
    返回选中的索引列表。

    要求：学术需求中 CIFAR-10/MNIST 均默认每类 50。
    """
    rng = np.random.default_rng(int(seed))
    targets_arr = np.array(targets)
    selected = []
    for cls in range(num_classes):
        cls_indices = np.where(targets_arr == cls)[0]
        rng.shuffle(cls_indices)
        chosen = cls_indices[:num_per_class].tolist()
        if len(chosen) < num_per_class:
            raise ValueError(f"Class {cls} has only {len(chosen)} samples, need {num_per_class}")
        selected.extend(chosen)
    rng.shuffle(selected)  # 打乱类间顺序
    return selected
```

3. **修改 `load_shieldfl_data()`**：
   - 支持 `dataset_name in ("CIFAR10", "MNIST")`
   - 根据 `dataset_name` 分派到 `_load_cifar10()` 或 `_load_mnist()`
   - `num_classes` 从数据集推断（两者都是 10）
   - `server_val_indices` 和 `server_trust_indices` 改用 `_stratified_balanced_sample()`
   - 参数新增 `val_per_class`（默认 50）和 `trust_per_class`（默认 50）
   - 保留旧的互斥断言：`server_val_set ∩ client_pool = ∅` 等

4. **更新 `ShieldFLDataAssets`**：新增 `num_classes` 字段。

**验证点**：
- 打印 val_loader 中各类样本计数，确认每类 == `val_per_class`
- 确认 MNIST 路径可下载和加载
- 互斥断言不变

---

### Step 2：新增 ResNet18 和 LeNet-5 模型

**新文件**：`model/resnet18.py`

```python
"""
CIFAR-10 适配的 ResNet18。
与标准 torchvision ResNet18 的区别：
- 第一层 conv: 3×3, stride=1, padding=1（替代 7×7, stride=2）
- 无 maxpool 层
- 适配 32×32 输入
"""
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.linear(out)
```

**新文件**：`model/lenet5.py`

```python
"""
LeNet-5：MNIST 28×28 单通道输入。
学术需求标记为 "LeNet-5 (SimpleCNN)"。
"""
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)   # 28→28
        self.conv2 = nn.Conv2d(6, 16, 5)              # 14→10, pool→5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)    # 28→14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)    # 14→10→5
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

**修改文件**：`model/model_hub.py`

```python
from .resnet18 import ResNet18
from .resnet20 import ResNet20
from .simple_cnn import SimpleCNN
from .lenet5 import LeNet5

MODEL_REGISTRY = {
    "SimpleCNN": SimpleCNN,
    "ResNet20": ResNet20,
    "ResNet18": ResNet18,
    "LeNet5": LeNet5,
}
```

**验证点**：
- `ResNet18()` 的 state_dict 键顺序稳定，含 BN 层
- `LeNet5()` 可接收 `(B, 1, 28, 28)` 输入
- `SimpleCNN` 保留不变（CIFAR-10 用）

---

### Step 3：新建 FedAvg 基线聚合器

**新文件**：`trainer/baseline_aggregator.py`

**设计意图**：M1（无攻击 FedAvg）、M2（攻击 + FedAvg）、M3（攻击 + 内置防御）实验需要一个**不经过 VeriFL 三阶段逻辑**的聚合路径。直接利用 `ServerAggregator` 基类的内置 `on_before_aggregation()` / `aggregate()` / `on_after_aggregation()` 钩子，这些钩子已经接入了 `FedMLAttacker` 和 `FedMLDefender`。

```python
"""
BaselineAggregator：用于 M1/M2/M3 的标准 FedAvg 聚合器。

不覆盖 on_before_aggregation / aggregate / on_after_aggregation，
因此 FedML 内置的 FedMLAttacker 模型攻击钩子和 FedMLDefender 防御钩子
会自动生效（由 ServerAggregator 基类调度）。

仅扩展 test() 以支持 ASR 评估。
"""
import copy, logging, time
from collections import OrderedDict
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn

from fedml.core import ServerAggregator


class BaselineAggregator(ServerAggregator):
    def __init__(self, model, args, data_assets=None, device=None):
        super().__init__(model, args)
        self.data_assets = data_assets
        self.device = device or torch.device("cpu")
        # ASR 评估参数
        self.eval_asr = bool(getattr(args, "eval_asr", False))
        self.trigger_size = int(getattr(args, "trigger_size", 3))
        self.trigger_value = float(getattr(args, "trigger_value", 1.0))
        self.target_label = int(getattr(args, "target_label", 0))
        # 聚合计时
        self._last_agg_time = 0.0

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters, strict=True)

    def aggregate(self, raw_client_model_or_grad_list):
        """带计时的聚合，委托给基类（FedAvg / FedMLDefender）。"""
        t0 = time.perf_counter()
        result = super().aggregate(raw_client_model_or_grad_list)
        self._last_agg_time = time.perf_counter() - t0
        logging.info("BaselineAggregator aggregate_time=%.4fs", self._last_agg_time)
        return result

    def test(self, test_data, device, args):
        """标准评估 + 可选 ASR 评估。"""
        self.device = device
        model = self.model
        model.to(device)
        model.eval()
        criterion = nn.CrossEntropyLoss().to(device)
        metrics: Dict[str, float] = {
            "test_correct": 0, "test_loss": 0.0, "test_total": 0,
            "test_accuracy": 0.0, "agg_time": self._last_agg_time,
        }
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

        # ASR 评估（仅在启用后门攻击时）
        if self.eval_asr and self.data_assets is not None:
            metrics.update(self._evaluate_asr(device))

        logging.info(
            "Baseline test | loss=%.6f | accuracy=%.4f | asr=%.4f | samples=%s",
            metrics["test_loss"], metrics["test_accuracy"],
            metrics.get("asr", -1.0), metrics["test_total"],
        )
        return metrics

    def _evaluate_asr(self, device) -> Dict[str, float]:
        """后门攻击成功率评估：在测试集上注入触发器，统计被分类为 target_label 的比例。"""
        model = self.model
        model.eval()
        total, success = 0, 0
        test_loader = self.data_assets.test_loader
        with torch.no_grad():
            for images, labels in test_loader:
                # 只对非 target_label 样本注入触发器
                mask = labels != self.target_label
                if mask.sum() == 0:
                    continue
                images_masked = images[mask].clone().to(device)
                # 注入触发器（右下角 patch）
                h, w = images_masked.shape[2], images_masked.shape[3]
                images_masked[:, :, h - self.trigger_size:, w - self.trigger_size:] = self.trigger_value
                logits = model(images_masked)
                _, predicted = torch.max(logits, 1)
                total += images_masked.size(0)
                success += (predicted == self.target_label).sum().item()
        asr = success / max(total, 1)
        return {"asr": asr, "asr_total": total, "asr_success": success}

    def test_all(self, train_data_local_dict, test_data_local_dict, device, args) -> bool:
        return False
```

**验证点**：
- 无攻击无防御时，`aggregate()` 走 `FedMLAggOperator.agg`（即标准 FedAvg）
- 启用 `enable_attack` 后，`on_before_aggregation()` 中 FedMLAttacker 钩子自动生效
- 启用 `enable_defense` 后，`aggregate()` 中 FedMLDefender 钩子自动生效
- ASR 评估可通过 `eval_asr: true` 配置项开关

---

### Step 4：修改 VeriFLAggregator——接入 FedMLAttacker 钩子 + 聚合计时

**文件**：`trainer/verifl_aggregator.py`

**改动 1**：`on_before_aggregation()` 中加入 FedMLAttacker 模型攻击钩子

```python
def on_before_aggregation(self, raw_client_model_or_grad_list):
    raw_client_model_or_grad_list = list(raw_client_model_or_grad_list)

    # ---- Phase 2 新增：FedMLAttacker 模型攻击钩子 ----
    from fedml.core.security.fedml_attacker import FedMLAttacker
    if FedMLAttacker.get_instance().is_model_attack():
        raw_client_model_or_grad_list = FedMLAttacker.get_instance().attack_model(
            raw_client_grad_list=raw_client_model_or_grad_list,
            extra_auxiliary_info=self.get_model_params(),
        )
    # ---- 钩子结束 ----

    client_idxs = [idx for idx in range(len(raw_client_model_or_grad_list))]
    logging.info(
        "VeriFL on_before_aggregation: %s client updates | attack_enabled=%s",
        len(client_idxs),
        FedMLAttacker.get_instance().is_attack_enabled(),
    )
    return raw_client_model_or_grad_list, client_idxs
```

**改动 2**：`aggregate()` 顶部和底部加入计时

```python
def aggregate(self, raw_client_model_or_grad_list):
    import time
    t0 = time.perf_counter()
    # ... 原有三阶段逻辑不变 ...
    self._last_agg_time = time.perf_counter() - t0
    logging.info("VeriFL aggregate_time=%.4fs", self._last_agg_time)
    return self._ndarrays_to_ordered_dict(final_params)
```

**改动 3**：`test()` 中加入 ASR 评估（复用与 BaselineAggregator 相同的 `_evaluate_asr` 逻辑）和 `agg_time` 指标

```python
def test(self, test_data, device, args):
    # ... 原有 MA/Loss 评估 ...
    metrics["agg_time"] = getattr(self, "_last_agg_time", 0.0)
    if bool(getattr(self.args, "eval_asr", False)) and self.data_assets is not None:
        metrics.update(self._evaluate_asr(device))
    return metrics
```

新增 `_evaluate_asr()` 方法，逻辑同 BaselineAggregator。

**验证点**：
- 当 `enable_attack: false` 时，VeriFL 聚合行为与 Phase 1 完全一致
- 当 `enable_attack: true, attack_type: "byzantine"` 时，`on_before_aggregation` 中恶意客户端模型被篡改后再进入 VeriFL 三阶段
- `agg_time` 可在日志和 metrics 中看到

---

### Step 5：修改 VeriFLTrainer——移除攻击硬拒绝，适配 FedML 内置攻击路径

**文件**：`trainer/verifl_trainer.py`

**改动概述**：

1. 移除 `train()` 中的 `NotImplementedError` 硬拒绝：

```python
def train(self, train_data, device, args):
    # Phase 2: 攻击注入由 FedML 内置机制处理
    # - label_flipping: 在 ClientTrainer.update_dataset() 中自动被 FedMLAttacker 调用
    # - byzantine / model_replacement: 在 ServerAggregator.on_before_aggregation() 中在服务端处理
    # 因此 train() 只需执行标准本地 SGD
    model = self.model
    model.to(device)
    model.train()
    # ... 后续训练代码不变 ...
```

2. 移除 `self.attack_type` 属性（攻击类型不再由 Trainer 判断）。Trainer 仅负责本地训练，攻击的扰动发生在 FedML 框架层。

**关键说明**：

FedML 三类攻击的注入点不同，VeriFLTrainer 的改动很小：

| 攻击 | 注入点 | Trainer 是否需要改动 |
|---|---|---|
| `label_flipping` | `ClientTrainer.update_dataset()` 基类方法 | 否（基类自动处理） |
| `byzantine` | `ServerAggregator.on_before_aggregation()` | 否（聚合器侧处理） |
| `model_replacement` | `ServerAggregator.on_before_aggregation()` | 否（聚合器侧处理） |

所以 Trainer 的核心改动就是**移除硬拒绝**。

---

### Step 6：修改 main 入口——支持聚合器类型切换

**文件**：`main_fedml_shieldfl.py`

**改动概述**：根据配置选择 `VeriFLAggregator` 或 `BaselineAggregator`。

```python
import fedml
from fedml import FedMLRunner

from data.data_loader import load_shieldfl_data
from model.model_hub import create_model
from trainer.verifl_aggregator import VeriFLAggregator
from trainer.baseline_aggregator import BaselineAggregator
from trainer.verifl_trainer import VeriFLTrainer
from utils.runtime import configure_runtime


if __name__ == "__main__":
    args = fedml.init(check_env=False)
    configure_runtime(args)

    device = fedml.device.get_device(args)
    args.device = device

    dataset, data_assets = load_shieldfl_data(args)
    model = create_model(args)

    trainer = VeriFLTrainer(model=model, args=args)

    # 聚合器选择：
    # - "verifl": VeriFL 三阶段聚合（VeriFL as defense）
    # - "fedavg" / 其他: 标准 FedAvg + FedML 内置攻防钩子
    aggregator_type = str(getattr(args, "aggregator_type", "verifl")).lower()
    if aggregator_type == "verifl":
        aggregator = VeriFLAggregator(
            model=model, args=args, data_assets=data_assets, device=device
        )
    else:
        aggregator = BaselineAggregator(
            model=model, args=args, data_assets=data_assets, device=device
        )

    fedml_runner = FedMLRunner(args, device, dataset, model, trainer, aggregator)
    fedml_runner.run()
```

**新增 YAML 字段**（在 `train_args` 或 `shieldfl_args` 下）：

```yaml
shieldfl_args:
  aggregator_type: "fedavg"   # "verifl" | "fedavg"
```

---

### Step 7：新建结构化指标采集模块

**新文件**：`eval/metrics.py`

**功能**：将每轮服务端测试结果写入 JSON Lines 文件，便于后续汇总和绘图。

```python
"""
结构化指标采集器。
每轮 append 一行 JSON 到指标文件，字段包括：
{
    "round": int,
    "test_accuracy": float,
    "test_loss": float,
    "asr": float | null,
    "agg_time": float,
    "model": str,
    "dataset": str,
    "attack_type": str,
    "defense_type": str,
    "aggregator_type": str,
    "pmr": float,
    "alpha": float,
    "seed": int,
    "hardware": str,      # "cpu" | "cuda:0" 等
    "gpu_accelerated": bool,
    "timestamp": str
}
"""
import json, os, logging
from datetime import datetime


class MetricsCollector:
    def __init__(self, output_dir: str, args):
        os.makedirs(output_dir, exist_ok=True)
        tag = f"{getattr(args, 'model', 'unknown')}_{getattr(args, 'dataset', 'unknown')}"
        tag += f"_atk-{getattr(args, 'attack_type', 'none')}"
        tag += f"_def-{getattr(args, 'defense_type', 'none')}"
        tag += f"_a{getattr(args, 'partition_alpha', 0.5)}"
        tag += f"_s{getattr(args, 'random_seed', 0)}"
        self.filepath = os.path.join(output_dir, f"metrics_{tag}.jsonl")
        self.args = args
        self._base_meta = {
            "model": str(getattr(args, "model", "")),
            "dataset": str(getattr(args, "dataset", "")),
            "attack_type": str(getattr(args, "attack_type", "none")),
            "defense_type": str(getattr(args, "defense_type", "none")),
            "aggregator_type": str(getattr(args, "aggregator_type", "fedavg")),
            "pmr": float(getattr(args, "ratio_of_poisoned_client", 0.0)),
            "alpha": float(getattr(args, "partition_alpha", 0.5)),
            "seed": int(getattr(args, "random_seed", 0)),
            "hardware": str(getattr(args, "device", "cpu")),
            "gpu_accelerated": bool(getattr(args, "using_gpu", False)),
        }

    def log_round(self, round_idx: int, metrics: dict):
        record = {**self._base_meta, "round": round_idx, "timestamp": datetime.now().isoformat()}
        record["test_accuracy"] = metrics.get("test_accuracy", None)
        record["test_loss"] = metrics.get("test_loss", None)
        record["asr"] = metrics.get("asr", None)
        record["agg_time"] = metrics.get("agg_time", None)
        with open(self.filepath, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logging.info("MetricsCollector round=%d -> %s", round_idx, self.filepath)
```

**接入方式**：在 `BaselineAggregator.test()` 和 `VeriFLAggregator.test()` 中，在日志输出之后调用 `MetricsCollector.log_round()`。或者在 `main` 入口中为 aggregator 注入 `MetricsCollector` 实例。

---

### Step 8：新建 ASR 评估模块

**新文件**：`eval/asr.py`

**功能**：独立的 ASR（Attack Success Rate）评估函数，可被两个聚合器共用。

```python
"""
后门攻击成功率 (ASR) 评估。

触发器注入规则：
- 在图片右下角 trigger_size × trigger_size 区域填充 trigger_value
- 仅对 label != target_label 的样本注入触发器
- ASR = 被模型预测为 target_label 的比例

使用场景：
- model_replacement_backdoor_attack 实验的 ASR 评估
"""
import torch
import torch.nn as nn
from typing import Dict


def evaluate_asr(
    model: nn.Module,
    test_loader,
    device: torch.device,
    target_label: int = 0,
    trigger_size: int = 3,
    trigger_value: float = 1.0,
) -> Dict[str, float]:
    model.eval()
    total, success = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            mask = labels != target_label
            if mask.sum() == 0:
                continue
            images_masked = images[mask].clone().to(device)
            h, w = images_masked.shape[2], images_masked.shape[3]
            images_masked[:, :, h - trigger_size:, w - trigger_size:] = trigger_value
            logits = model(images_masked)
            _, predicted = torch.max(logits, 1)
            total += images_masked.size(0)
            success += (predicted == target_label).sum().item()
    asr = success / max(total, 1)
    return {"asr": asr, "asr_total": total, "asr_success": success}
```

---

### Step 9：处理 Bulyan 防御的接入问题

**问题**：`BulyanDefense` 存在于 bulyan_defense.py 但未在 `FedMLDefender` 中注册。且其接口为 `run()` 而非 `defend_before_aggregation()` / `defend_on_aggregation()`。

**两种解决方案**（任选一种）：

**方案 A：最小侵入式——在 BaselineAggregator 中手动调用**

在 `BaselineAggregator` 中增加 Bulyan 特殊路径：

```python
def aggregate(self, raw_client_model_or_grad_list):
    if str(getattr(self.args, "defense_type", "")).strip() == "bulyan":
        from fedml.core.security.defense.bulyan_defense import BulyanDefense
        bulyan = BulyanDefense(self.args)
        return bulyan.run(
            raw_client_model_or_grad_list,
            base_aggregation_func=None,
            extra_auxiliary_info=self.get_model_params(),
        )
    t0 = time.perf_counter()
    result = super().aggregate(raw_client_model_or_grad_list)
    self._last_agg_time = time.perf_counter() - t0
    return result
```

**方案 B：小补丁注册到 FedMLDefender**

在 `main_fedml_shieldfl.py` 启动时 monkey-patch 注册：

```python
from fedml.core.security.fedml_defender import FedMLDefender
from fedml.core.security.defense.bulyan_defense import BulyanDefense

_original_init = FedMLDefender.init
def _patched_init(self, args):
    _original_init(self, args)
    if hasattr(args, "defense_type") and args.defense_type.strip() == "bulyan":
        self.is_enabled = True
        self.defense_type = "bulyan"
        self.defender = BulyanDefense(args)
FedMLDefender.init = _patched_init
```

**推荐方案 A**：更简单、更可控，不修改 FedML 全局状态。

**约束**：Bulyan 要求 `n ≥ 4f + 3`。当 `byzantine_client_num=1` 时，至少需要 7 客户端。CPU 轻量测试时需注意此约束。

---

### Step 10：更新运行时模块——预留 single-gpu-deterministic

**文件**：`utils/runtime.py`

**改动概述**：

```python
def configure_runtime(args):
    runtime_mode = getattr(args, "runtime_mode", "cpu-deterministic")
    # ... 原有 seed 设置不变 ...

    if runtime_mode == "single-gpu-deterministic":
        # 预留：GPU 可用时启用
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
            logging.info("Runtime: single-gpu-deterministic enabled (GPU)")
        else:
            logging.warning(
                "Runtime: single-gpu-deterministic requested but no GPU available, "
                "falling back to cpu-deterministic"
            )
    # ... 打印设备/配置摘要 ...
    _print_runtime_summary(args, runtime_mode, seed)


def _print_runtime_summary(args, runtime_mode, seed):
    """启动时打印运行时上下文摘要，便于结果追溯。"""
    info = {
        "runtime_mode": runtime_mode,
        "seed": seed,
        "device": str(getattr(args, "device", "N/A")),
        "using_gpu": bool(getattr(args, "using_gpu", False)),
        "cuda_available": torch.cuda.is_available(),
        "model": getattr(args, "model", "N/A"),
        "dataset": getattr(args, "dataset", "N/A"),
        "attack_type": getattr(args, "attack_type", "none"),
        "defense_type": getattr(args, "defense_type", "none"),
        "aggregator_type": getattr(args, "aggregator_type", "verifl"),
    }
    logging.info("=== ShieldFL Runtime Context ===")
    for k, v in info.items():
        logging.info("  %s: %s", k, v)
```

---

### Step 11：新建实验编排脚本

**新文件**：`scripts/run_experiment.sh`

**功能**：根据参数组合生成 YAML 并运行 mpirun，适配 CPU 轻量模式。

```bash
#!/usr/bin/env bash
# 使用方法：
#   bash scripts/run_experiment.sh \
#     --model ResNet18 --dataset cifar10 \
#     --attack byzantine --defense none --aggregator fedavg \
#     --pmr 0.2 --alpha 0.5 --seed 0 \
#     --rounds 3 --clients 5 --epochs 1 --batch_size 32

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# 参数解析
MODEL="SimpleCNN"; DATASET="cifar10"; ATTACK="none"; DEFENSE="none"
AGGREGATOR="fedavg"; PMR="0.0"; ALPHA="0.5"; SEED="0"
ROUNDS="3"; CLIENTS="3"; EPOCHS="1"; BATCH="32"

while [[ $# -gt 0 ]]; do
  case $1 in
    --model) MODEL="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --attack) ATTACK="$2"; shift 2 ;;
    --defense) DEFENSE="$2"; shift 2 ;;
    --aggregator) AGGREGATOR="$2"; shift 2 ;;
    --pmr) PMR="$2"; shift 2 ;;
    --alpha) ALPHA="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --rounds) ROUNDS="$2"; shift 2 ;;
    --clients) CLIENTS="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch_size) BATCH="$2"; shift 2 ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

# 计算攻击参数
ENABLE_ATTACK="false"
ATTACK_TYPE="none"
BYZANTINE_NUM=0
if [[ "$ATTACK" != "none" ]]; then
  ENABLE_ATTACK="true"
  ATTACK_TYPE="$ATTACK"
  BYZANTINE_NUM=$(python3 -c "import math; print(max(1, math.ceil($CLIENTS * $PMR)))")
fi

ENABLE_DEFENSE="false"
DEFENSE_TYPE="none"
if [[ "$DEFENSE" != "none" ]]; then
  ENABLE_DEFENSE="true"
  DEFENSE_TYPE="$DEFENSE"
fi

# 生成临时配置
CONFIG_FILE="/tmp/shieldfl_exp_${MODEL}_${DATASET}_${ATTACK}_${DEFENSE}_a${ALPHA}_s${SEED}.yaml"
WORKER_NUM=$CLIENTS

cat > "$CONFIG_FILE" <<EOF
common_args:
  training_type: "cross_silo"
  random_seed: ${SEED}

data_args:
  dataset: "${DATASET}"
  data_cache_dir: ${SCRIPT_DIR}/data
  partition_method: "hetero"
  partition_alpha: ${ALPHA}
  server_val_size: 500
  server_trust_size: 500
  val_per_class: 50
  trust_per_class: 50
  client_pool_max_size: 5000
  max_samples_per_client: 500
  test_subset_size: 1000
  num_workers: 0

model_args:
  model: "${MODEL}"

train_args:
  federated_optimizer: "FedAvg"
  client_id_list:
  client_num_in_total: ${CLIENTS}
  client_num_per_round: ${CLIENTS}
  comm_round: ${ROUNDS}
  epochs: ${EPOCHS}
  batch_size: ${BATCH}
  client_optimizer: sgd
  learning_rate: 0.01
  weight_decay: 0.0
  momentum: 0.9

  enable_attack: ${ENABLE_ATTACK}
  attack_type: "${ATTACK_TYPE}"
  byzantine_client_num: ${BYZANTINE_NUM}
  attack_mode: "flip"
  original_class_list: [0,1,2,3,4,5,6,7,8,9]
  target_class_list: [9,8,7,6,5,4,3,2,1,0]
  ratio_of_poisoned_client: ${PMR}

  enable_defense: ${ENABLE_DEFENSE}
  defense_type: "${DEFENSE_TYPE}"
  byzantine_client_num: ${BYZANTINE_NUM}

  server_momentum: 0.9
  server_lr: 0.3
  pop_size: 15
  generations: 10
  lambda_reg: 0.01
  cpu_transfer: true

  eval_asr: $([ "$ATTACK" = "model_replacement" ] && echo "true" || echo "false")
  target_label: 0
  trigger_size: 3
  trigger_value: 1.0

validation_args:
  frequency_of_the_test: 1

device_args:
  worker_num: ${WORKER_NUM}
  using_gpu: false
  gpu_mapping_file: config/gpu_mapping.yaml
  gpu_mapping_key: mapping_default

comm_args:
  backend: "MPI"
  is_mobile: 0

tracking_args:
  log_file_dir: ./log
  enable_wandb: false
  using_mlops: false

shieldfl_args:
  runtime_mode: "cpu-deterministic"
  enforce_determinism: true
  sort_client_updates: true
  aggregator_type: "${AGGREGATOR}"
  metrics_output_dir: "./results"
EOF

echo "=== Running: model=${MODEL} dataset=${DATASET} attack=${ATTACK} defense=${DEFENSE} pmr=${PMR} alpha=${ALPHA} seed=${SEED} ==="
cd "$SCRIPT_DIR"
TOTAL_PROC=$((WORKER_NUM + 1))
mpirun -np $TOTAL_PROC python main_fedml_shieldfl.py --cf "$CONFIG_FILE"
```

**新文件**：`scripts/run_m1_baseline_cpu.sh`

```bash
#!/usr/bin/env bash
# M1：基线可信 CPU smoke test
# 轻量配置：3 轮，3 客户端
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

for MODEL in "SimpleCNN" "ResNet18"; do
  if [ "$MODEL" = "SimpleCNN" ]; then DATASET="mnist"; else DATASET="cifar10"; fi
  for ALPHA in 0.5 100; do
    for SEED in 0 1 2; do
      bash "$DIR/run_experiment.sh" \
        --model "$MODEL" --dataset "$DATASET" \
        --attack none --defense none --aggregator fedavg \
        --pmr 0.0 --alpha "$ALPHA" --seed "$SEED" \
        --rounds 3 --clients 3 --epochs 1 --batch_size 32
    done
  done
done
```

**新文件**：`scripts/run_m2_attacks_cpu.sh`

```bash
#!/usr/bin/env bash
# M2：攻击生效 CPU smoke test
# 轻量配置：3 轮，5 客户端（PMR=20% → 1 恶意）
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

for MODEL in "SimpleCNN" "ResNet18"; do
  if [ "$MODEL" = "SimpleCNN" ]; then DATASET="mnist"; else DATASET="cifar10"; fi
  for ATTACK in "byzantine" "label_flipping" "model_replacement"; do
    for ALPHA in 0.5; do
      for SEED in 0; do
        bash "$DIR/run_experiment.sh" \
          --model "$MODEL" --dataset "$DATASET" \
          --attack "$ATTACK" --defense none --aggregator fedavg \
          --pmr 0.2 --alpha "$ALPHA" --seed "$SEED" \
          --rounds 3 --clients 5 --epochs 1 --batch_size 32
      done
    done
  done
done
```

---

### Step 12：新增配置文件模板

以下配置文件置于 `config/` 目录。

**`config/fedml_config_m1_cifar10_cpu.yaml`**：M1 基线 ResNet18 + CIFAR10

```yaml
common_args:
  training_type: "cross_silo"
  random_seed: 0

data_args:
  dataset: "cifar10"
  data_cache_dir: ./data
  partition_method: "hetero"
  partition_alpha: 0.5
  val_per_class: 50
  trust_per_class: 50
  client_pool_max_size: 5000
  max_samples_per_client: 500
  test_subset_size: 1000
  num_workers: 0

model_args:
  model: "ResNet18"

train_args:
  federated_optimizer: "FedAvg"
  client_id_list:
  client_num_in_total: 3
  client_num_per_round: 3
  comm_round: 5
  epochs: 1
  batch_size: 32
  client_optimizer: sgd
  learning_rate: 0.01
  weight_decay: 0.0
  momentum: 0.9
  enable_attack: false
  attack_type: "none"
  enable_defense: false
  defense_type: "none"

validation_args:
  frequency_of_the_test: 1

device_args:
  worker_num: 3
  using_gpu: false
  gpu_mapping_file: config/gpu_mapping.yaml
  gpu_mapping_key: mapping_default

comm_args:
  backend: "MPI"
  is_mobile: 0

tracking_args:
  log_file_dir: ./log
  enable_wandb: false

shieldfl_args:
  runtime_mode: "cpu-deterministic"
  enforce_determinism: true
  sort_client_updates: true
  aggregator_type: "fedavg"
  metrics_output_dir: "./results"
```

**`config/fedml_config_m2_byzantine_cpu.yaml`**：M2 Byzantine 攻击（flip 模式）

```yaml
common_args:
  training_type: "cross_silo"
  random_seed: 0

data_args:
  dataset: "cifar10"
  data_cache_dir: ./data
  partition_method: "hetero"
  partition_alpha: 0.5
  val_per_class: 50
  trust_per_class: 50
  client_pool_max_size: 5000
  max_samples_per_client: 500
  test_subset_size: 1000
  num_workers: 0

model_args:
  model: "ResNet18"

train_args:
  federated_optimizer: "FedAvg"
  client_id_list:
  client_num_in_total: 5
  client_num_per_round: 5
  comm_round: 5
  epochs: 1
  batch_size: 32
  client_optimizer: sgd
  learning_rate: 0.01
  weight_decay: 0.0
  momentum: 0.9
  enable_attack: true
  attack_type: "byzantine"
  byzantine_client_num: 1
  attack_mode: "flip"
  enable_defense: false
  defense_type: "none"

validation_args:
  frequency_of_the_test: 1

device_args:
  worker_num: 5
  using_gpu: false
  gpu_mapping_file: config/gpu_mapping.yaml
  gpu_mapping_key: mapping_default

comm_args:
  backend: "MPI"
  is_mobile: 0

shieldfl_args:
  runtime_mode: "cpu-deterministic"
  aggregator_type: "fedavg"
  metrics_output_dir: "./results"
```

**`config/fedml_config_m3_verifl_vs_attack_cpu.yaml`**：M3 VeriFL 防御 × Byzantine 攻击

```yaml
# M3: VeriFL 作为防御 vs Byzantine 攻击
common_args:
  training_type: "cross_silo"
  random_seed: 0

data_args:
  dataset: "cifar10"
  data_cache_dir: ./data
  partition_method: "hetero"
  partition_alpha: 0.5
  val_per_class: 50
  trust_per_class: 50
  client_pool_max_size: 5000
  max_samples_per_client: 500
  test_subset_size: 1000
  num_workers: 0

model_args:
  model: "ResNet18"

train_args:
  federated_optimizer: "FedAvg"
  client_id_list:
  client_num_in_total: 5
  client_num_per_round: 5
  comm_round: 3
  epochs: 1
  batch_size: 32
  client_optimizer: sgd
  learning_rate: 0.01
  weight_decay: 0.0
  momentum: 0.9

  # 攻击由 FedMLAttacker 执行
  enable_attack: true
  attack_type: "byzantine"
  byzantine_client_num: 1
  attack_mode: "flip"

  # VeriFL IS the defense → 不启用 FedML 内置防御
  enable_defense: false
  defense_type: "none"

  # VeriFL 超参
  server_momentum: 0.9
  server_lr: 0.3
  pop_size: 15
  generations: 10
  lambda_reg: 0.01

validation_args:
  frequency_of_the_test: 1

device_args:
  worker_num: 5
  using_gpu: false
  gpu_mapping_file: config/gpu_mapping.yaml
  gpu_mapping_key: mapping_default

comm_args:
  backend: "MPI"

shieldfl_args:
  runtime_mode: "cpu-deterministic"
  aggregator_type: "verifl"      # <--- VeriFL 聚合
  metrics_output_dir: "./results"
```

---

### Step 13：更新 gpu_mapping.yaml 以支持 5/7 客户端配置

**文件**：`config/gpu_mapping.yaml`

需新增适配不同客户端数量的映射条目：

```yaml
mapping_default:
  MPI_HOST_0:
    - host1
  MPI_HOST_1:
    - host1
  MPI_HOST_2:
    - host1
  MPI_HOST_3:
    - host1

mapping_5clients:
  MPI_HOST_0:
    - host1
  MPI_HOST_1:
    - host1
  MPI_HOST_2:
    - host1
  MPI_HOST_3:
    - host1
  MPI_HOST_4:
    - host1
  MPI_HOST_5:
    - host1

mapping_7clients:
  MPI_HOST_0:
    - host1
  MPI_HOST_1:
    - host1
  MPI_HOST_2:
    - host1
  MPI_HOST_3:
    - host1
  MPI_HOST_4:
    - host1
  MPI_HOST_5:
    - host1
  MPI_HOST_6:
    - host1
  MPI_HOST_7:
    - host1
```

---

### Step 14：BN 校准与 server momentum 跨轮一致性验证

**目标**（对应 MOVE P1 第 5 项）：在有攻击的场景下，确认 VeriFL 聚合器的 BN 校准和 server momentum 状态不被攻击破坏。

**验证方法**：

1. 运行 `config/fedml_config_m3_verifl_vs_attack_cpu.yaml`（ResNet18 含 BN 层）
2. 在日志中确认以下条目出现：
   - `VeriFL phase-3 complete | server_momentum lr=... momentum=...`（每轮）
   - `VeriFL bn_recalibration complete | enabled=True`（每轮）
3. 确认 `global_model_buffer` 和 `velocity_buffer` 在跨轮间连续更新（通过添加诊断日志）

**改动**：在 `VeriFLAggregator.aggregate()` 中增加诊断日志（可通过配置项控制是否输出）：

```python
if bool(getattr(self.args, "debug_state_tracking", False)):
    gb_norm = np.sqrt(sum(np.sum(x**2) for x in self.global_model_buffer))
    vb_norm = np.sqrt(sum(np.sum(x**2) for x in self.velocity_buffer))
    logging.info("VeriFL state_check | global_buffer_norm=%.6f | velocity_buffer_norm=%.6f", gb_norm, vb_norm)
```

---

## 6. CPU 轻量测试策略

### 6.1 原则

- CPU smoke test 目标是**验证管道正确性**，不是达到学术精度门槛
- 学术精度门槛（M1: MA ≥ 85% 等）需 50–100 轮 GPU 训练，不属于 Phase 2 CPU 验收范围
- CPU smoke test 只需证明：攻击被执行、防御被调用、指标被正确采集

### 6.2 CPU smoke test 参数建议

| 参数 | M1 基线 | M2 攻击 | M3 防御 |
|---|---|---|---|
| `comm_round` | 3–5 | 3–5 | 3 |
| `client_num_in_total` | 3 | 5 | 5（Bulyan 需 7+） |
| `epochs` | 1 | 1 | 1 |
| `batch_size` | 32 | 32 | 32 |
| `max_samples_per_client` | 300–500 | 300–500 | 300–500 |
| `val_per_class` | 50 | 50 | 50 |
| `test_subset_size` | 500–1000 | 500–1000 | 500–1000 |
| `pop_size`（VeriFL 时）| N/A | N/A | 15 |
| `generations`（VeriFL 时）| N/A | N/A | 10 |

### 6.3 CPU smoke test 验收标准（不等同于学术验收）

| 检查项 | 通过标准 |
|---|---|
| M1 FedAvg 无攻击 | 3 轮后 loss 下降，accuracy > random guess |
| M2 Byzantine (flip) | 日志中出现 `byzantine_idxs`，accuracy 低于无攻击基线 |
| M2 Label Flipping | 日志中出现 data poisoning，accuracy 低于无攻击基线 |
| M2 Model Replacement | 日志中出现 malicious_idx，eval_asr 指标被输出 |
| M3 Krum | 日志中出现 krum scores 计算 |
| M3 VeriFL + 攻击 | 三阶段聚合 + BN 校准日志正常出现 |
| 指标采集 | `results/` 目录下生成 `.jsonl` 文件 |
| 两条任务线 | ResNet18+CIFAR10 和 LeNet5+MNIST 均可运行 |

---

## 7. FedML 攻击/防御参数速查表

### 7.1 攻击参数

| 攻击类型 | `attack_type` 值 | 关键参数 | 说明 |
|---|---|---|---|
| Byzantine (flip) | `"byzantine"` | `byzantine_client_num`, `attack_mode: "flip"` | 服务端模型攻击；flip 模式对选中客户端做 Δ 翻转 |
| Label Flipping | `"label_flipping"` | `original_class_list`, `target_class_list`, `ratio_of_poisoned_client` | 客户端数据投毒；label → `(num_classes - 1 - label)` |
| Model Replacement | `"model_replacement"` | `malicious_client_id` (可选), `scale_factor_S` (可选) | 服务端模型攻击；gamma 缩放替换 |

### 7.2 防御参数

| 防御类型 | `defense_type` 值 | 关键参数 | 注意事项 |
|---|---|---|---|
| Krum | `"krum"` | `byzantine_client_num`, `krum_param_m` | 要求 `2f + 2 ≤ n - m` |
| RFA | `"rfa"` | （无额外参数） | 几何中位数聚合 |
| Trimmed Mean | `"trimmed_mean"` | `beta`（裁剪比例） | `beta ∈ [0, 0.5)` |
| CClip | `"cclip"` | `tau`, `bucket_size` | 分桶+裁剪 |
| Bulyan | `"bulyan"` | `byzantine_client_num`, `client_num_per_round` | **未注册在 FedMLDefender**；要求 `n ≥ 4f + 3` |
| VeriFL | 不走 FedMLDefender | `aggregator_type: "verifl"` + VeriFL 超参 | 通过聚合器切换实现 |

---

## 8. 文件变更总览

### 8.1 修改文件

| 文件 | 改动类型 | 改动摘要 |
|---|---|---|
| `data/data_loader.py` | 重大改动 | + MNIST 支持；+ 分层均衡采样；保持 FedML 8-slot tuple 输出 |
| `model/model_hub.py` | 小改动 | + ResNet18/LeNet5 注册 |
| `trainer/verifl_trainer.py` | 小改动 | 移除 `NotImplementedError` 硬拒绝 |
| `trainer/verifl_aggregator.py` | 中等改动 | + FedMLAttacker 钩子；+ 聚合计时；+ ASR 评估；+ 状态诊断日志 |
| `main_fedml_shieldfl.py` | 中等改动 | + 聚合器类型切换；+ MetricsCollector 初始化 |
| `utils/runtime.py` | 小改动 | + single-gpu-deterministic 预留；+ 运行时摘要打印 |
| `config/gpu_mapping.yaml` | 小改动 | + 5/7 客户端映射 |

### 8.2 新增文件

| 文件 | 说明 |
|---|---|
| `model/resnet18.py` | CIFAR-10 适配的 ResNet18 |
| `model/lenet5.py` | MNIST 适配的 LeNet-5 |
| `trainer/baseline_aggregator.py` | FedAvg 基线聚合器（内置攻防钩子透传） |
| `eval/__init__.py` | 空 |
| `eval/metrics.py` | 结构化指标采集器（JSONL 输出） |
| `eval/asr.py` | ASR 评估函数 |
| `scripts/run_experiment.sh` | 参数化实验编排脚本 |
| `scripts/run_m1_baseline_cpu.sh` | M1 基线 CPU smoke test |
| `scripts/run_m2_attacks_cpu.sh` | M2 攻击 CPU smoke test |
| `config/fedml_config_m1_cifar10_cpu.yaml` | M1 配置模板 |
| `config/fedml_config_m1_mnist_cpu.yaml` | M1 MNIST 配置模板 |
| `config/fedml_config_m2_byzantine_cpu.yaml` | M2 Byzantine 配置 |
| `config/fedml_config_m2_label_flipping_cpu.yaml` | M2 Label Flipping 配置 |
| `config/fedml_config_m2_model_replacement_cpu.yaml` | M2 Model Replacement 配置 |
| `config/fedml_config_m3_verifl_vs_attack_cpu.yaml` | M3 VeriFL 防御配置 |
| `config/fedml_config_m3_krum_cpu.yaml` | M3 Krum 防御配置 |

---

## 9. Phase 2 实施顺序（压缩版）

| 序号 | 步骤 | 可验证输出 |
|---|---|---|
| 1 | 升级 data_loader：MNIST + 分层均衡采样 | 打印 val 各类计数，MNIST 可加载 |
| 2 | 新增 ResNet18 + LeNet5 | `create_model(args)` 创建正确模型 |
| 3 | 新建 BaselineAggregator | 无攻击 FedAvg 可跑通 |
| 4 | 修改 main 入口支持聚合器切换 | `aggregator_type: "fedavg"` 时走 baseline |
| 5 | 修改 VeriFLTrainer 移除硬拒绝 | `attack_type != "none"` 不再报错 |
| 6 | 修改 VeriFLAggregator 接入攻击钩子 + 计时 | VeriFL + byzantine 日志显示攻击被执行 |
| 7 | 新建 eval 模块（metrics + asr） | JSONL 文件生成 |
| 8 | 新增配置文件模板 | 各配置可被 mpirun 直接使用 |
| 9 | **M1 CPU smoke test** | FedAvg 基线两条任务线跑通 |
| 10 | **M2 CPU smoke test** | 三类攻击在两条任务线上跑通 |
| 11 | **M3 基础设施验证** | Krum/VeriFL + 攻击 CPU smoke test 跑通 |
| 12 | BN/momentum 一致性验证 | ResNet18 + VeriFL + 攻击 下三阶段日志完整 |
| 13 | 新建实验编排脚本 | 一键运行多组合 |
| 14 | 运行时模块更新 | 运行时摘要打印 |

---

## 10. Phase 2 验收标准

### 10.1 数据层

- [ ] MNIST 数据可正确加载，data_loader.py 支持 `dataset: "mnist"` 和 `dataset: "cifar10"`
- [ ] server_val 和 server_trust 采用分层均衡采样，每类样本数相等
- [ ] 互斥断言保持：`server_val_set ∩ client_pool = ∅` 等

### 10.2 模型层

- [ ] `ResNet18` 可在 CIFAR-10 上前向传播
- [ ] `LeNet5` 可在 MNIST 上前向传播
- [ ] 两个模型的 state_dict 键顺序稳定

### 10.3 攻击链路

- [ ] `byzantine_attack`（flip 模式）在 FedAvg 聚合下可运行，日志中显示 `byzantine_idxs`
- [ ] `label_flipping_attack` 在 FedAvg 聚合下可运行，日志中显示数据投毒
- [ ] `model_replacement_backdoor_attack` 在 FedAvg 聚合下可运行
- [ ] 三类攻击在 VeriFL 聚合器下均可运行（`on_before_aggregation` 中 FedMLAttacker 钩子生效）

### 10.4 防御链路

- [ ] `krum` 防御在 BaselineAggregator 下可运行
- [ ] `rfa` 防御在 BaselineAggregator 下可运行
- [ ] `trimmed_mean` 防御在 BaselineAggregator 下可运行
- [ ] `cclip` 防御在 BaselineAggregator 下可运行
- [ ] Bulyan 防御可通过手动调用路径工作（需足够客户端数量）
- [ ] VeriFL 聚合器作为防御与 FedML 内置攻击协同工作

### 10.5 ASR 评估

- [ ] `model_replacement` 攻击场景下输出 ASR 指标
- [ ] ASR 评估函数可正确注入触发器并统计成功率

### 10.6 指标采集

- [ ] 每轮指标以 JSONL 格式写入 `results/` 目录
- [ ] 指标包含：round, test_accuracy, test_loss, asr, agg_time
- [ ] 指标文件包含实验元数据：model, dataset, attack_type, defense_type, alpha, seed

### 10.7 两条任务线

- [ ] `ResNet18 + CIFAR10` 在 CPU 上无攻击跑通 3 轮，accuracy > random guess
- [ ] `LeNet5 + MNIST` 在 CPU 上无攻击跑通 3 轮，accuracy > random guess
- [ ] 两条任务线均可被攻击场景正常运行

### 10.8 BN/momentum 一致性

- [ ] `ResNet18`（含 BN）在 VeriFL + 攻击场景下，`bn_recalibration` 每轮触发
- [ ] `global_model_buffer` 和 `velocity_buffer` 在跨轮间连续更新

### 10.9 实验编排

- [ ] `scripts/run_experiment.sh` 可根据参数生成配置并运行
- [ ] `scripts/run_m1_baseline_cpu.sh` 可一键运行 M1 smoke test
- [ ] `scripts/run_m2_attacks_cpu.sh` 可一键运行 M2 smoke test

---

## 11. Phase 2 明确不做的事

以下内容不属于 Phase 2 范围：

- 学术精度门槛验证（M1 的 MA ≥ 85% 等需要 50–100 轮 GPU 训练）
- 完整的 PMR × α × seed 全组合实验（留到有 GPU 资源后）
- `mean ± std` 汇总统计与自动图表生成（M4 范围）
- GPU 大规模对照实验
- cross-cloud 部署
- wandb / MLOps 集成
- DDP / DataParallel
- secure aggregation / DP 兼容性实验
- full attack menu 迁移（ShieldFL 自有的 adaptive_backdoor / pure_scaling 等）
- 阈值校准干跑（dry-run）——需要有 GPU 和足够时间
- FedML 核心库修改（Bulyan 注册以外的改动）

---

## 12. 已知风险与应对

| 风险 | 说明 | 应对 |
|---|---|---|
| FedML label_flipping 参数不直观 | 需要 `original_class_list` + `target_class_list`，不是简单 `num_classes` | 在配置模板中写好默认值 |
| Byzantine attack 选择客户端是随机的 | FedML 内置 `byzantine_attack` 使用 `sample_some_clients` 随机选取恶意客户端 | 固定 seed 可保证复现性 |
| Bulyan 客户端数量约束 | `n ≥ 4f + 3`，CPU 轻量测试时 client 数可能不够 | Bulyan smoke test 用 7 客户端 |
| MNIST 下载可能失败 | torchvision MNIST 下载源不稳定 | 提供离线数据准备方案 |
| CPU 下 MPI 进程过多导致内存不足 | 7+ 进程各自加载模型和数据 | 限制 max_samples_per_client 和 test_subset_size |
| FedMLAttacker 是全局单例 | 所有 MPI 进程共享同一初始化 | 确保 YAML 中攻击参数一致传递到所有 rank |
| VeriFL 聚合器中 FedMLDefender 不应同时启用 | 双重防御会污染实验结论 | 配置中当 `aggregator_type: "verifl"` 时强制 `enable_defense: false` |

---

## 13. Phase 2 最终交付物

Phase 2 结束时，仓库中应新增或更新以下内容：

### 新增文件
- `model/resnet18.py`
- `model/lenet5.py`
- `trainer/baseline_aggregator.py`
- `eval/__init__.py`
- `eval/metrics.py`
- `eval/asr.py`
- `scripts/run_experiment.sh`
- `scripts/run_m1_baseline_cpu.sh`
- `scripts/run_m2_attacks_cpu.sh`
- `config/fedml_config_m1_cifar10_cpu.yaml`
- `config/fedml_config_m1_mnist_cpu.yaml`
- `config/fedml_config_m2_byzantine_cpu.yaml`
- `config/fedml_config_m2_label_flipping_cpu.yaml`
- `config/fedml_config_m2_model_replacement_cpu.yaml`
- `config/fedml_config_m3_verifl_vs_attack_cpu.yaml`
- `config/fedml_config_m3_krum_cpu.yaml`
- `RUN_RECORD_PHASE2.md`（运行记录）

### 更新文件
- `data/data_loader.py`
- `model/model_hub.py`
- `trainer/verifl_trainer.py`
- `trainer/verifl_aggregator.py`
- `main_fedml_shieldfl.py`
- `utils/runtime.py`
- `config/gpu_mapping.yaml`

### 运行记录
一份 `RUN_RECORD_PHASE2.md`，记录：
- 每条 smoke test 的配置、日志摘要、关键证据截取
- 哪些攻击/防御路径已在 CPU 上跑通
- BN/momentum 一致性验证结论
- 已知局限和 GPU 后续计划

---

## 14. 一句话版 Phase 2

> **在 Phase 1 已验证的 VeriFL 宿主之上，补齐两条任务线（ResNet18+CIFAR10、LeNet5+MNIST）的数据/模型/攻击/防御/指标全链路，使项目达到"可在任意 attack × defense × PMR × α × seed 组合上一键运行"的学术基础设施状态，并在纯 CPU 上跑通覆盖 M1/M2/M3 的轻量 smoke test。**