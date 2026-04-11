# 可插拔防御注册表 & VeriFL → v16 重命名 — 变更文档

> **日期**：2026-04-09  
> **范围**：`fedml_defender.py` 核心防御框架 + ShieldFL 项目文件重命名  
> **影响**：FedML 安全防御子系统、ShieldFL 实验项目

---

## 1. 变更概览

| # | 变更 | 类型 | 涉及文件 |
|---|------|------|----------|
| C-1 | `FedMLDefender` 从 if-elif 硬编码改为注册表模式 | 架构重构 | `fedml_defender.py` |
| C-2 | `VeriFLAggregator` → `VeriFLv16Aggregator` | 重命名 | `verifl_v16_aggregator.py` |
| C-3 | `VeriFLTrainer` → `VeriFLv16Trainer` | 重命名 | `verifl_v16_trainer.py` |
| C-4 | 入口文件和包导出更新 | 适配 | `main_fedml_shieldfl.py`, `__init__.py` |

---

## 2. C-1：可插拔防御注册表

### 2.1 改动原因

原 `FedMLDefender.init()` 通过 **17 个 if-elif 分支**硬编码所有防御算法的实例化逻辑，同时 `is_defense_before_aggregation()` 等方法通过硬编码列表判断阶段。这导致：

1. **不可扩展** — 添加新防御必须修改 `fedml_defender.py` 核心文件
2. **状态泄漏风险** — `init()` 未显式重置旧状态，连续实验可能产生交叉污染
3. **维护负担** — 阶段判断列表与 init 分支经常不同步

### 2.2 新架构

```
┌─ _DefenseRegistry (模块级单例) ──────────────────────────────┐
│                                                              │
│  register_lazy(type, module, class, phases, aliases)         │
│    → 延迟导入：仅在首次 get() 时才 import 防御模块           │
│                                                              │
│  register(type, cls, phases, aliases)                        │
│    → 立即注册：用于外部自定义防御                              │
│                                                              │
│  get(type) → _DefenseEntry(cls, phases)                      │
│    → 查找并返回防御类 + 阶段元数据                            │
│                                                              │
│  available() → List[str]                                     │
│    → 列出所有已注册防御类型                                    │
└──────────────────────────────────────────────────────────────┘

┌─ FedMLDefender (运行时单例) ─────────────────────────────────┐
│                                                              │
│  init(args):                                                 │
│    ① 显式重置 defender / defense_type / _phases              │
│    ② 从注册表 get() → 获取 cls + phases                      │
│    ③ cls(args) 创建新实例                                    │
│                                                              │
│  is_defense_before_aggregation():                            │
│    → PHASE_BEFORE in self._phases  (不再硬编码列表)           │
│                                                              │
│  register_defense(type, cls, phases):                        │
│    → 公开 API，外部代码零侵入注册自定义防御                    │
│                                                              │
│  available_defenses():                                       │
│    → 公开 API，查询所有可用防御                               │
└──────────────────────────────────────────────────────────────┘
```

### 2.3 关键设计决策

| 决策 | 理由 |
|------|------|
| **延迟导入** | 16 个内置防御通过 `register_lazy()` 按需加载，避免 import 时加载全部 17 个模块 |
| **阶段声明式** | 每个防御在注册时声明参与的 hook 阶段（`before_aggregation` / `on_aggregation` / `after_aggregation`），`is_defense_*()` 方法直接查询该元数据 |
| **显式状态重置** | `init()` 开头强制 `self.defender = None; self.defense_type = None; self._phases = frozenset()`，防止跨实验状态泄漏 |
| **别名支持** | `krum` 和 `multikrum` 共享同一个 `KrumDefense` 类（通过 aliases 参数） |

### 2.4 内置防御阶段映射表

| defense_type | 类 | before | on | after |
|---|---|:---:|:---:|:---:|
| `norm_diff_clipping` | NormDiffClippingDefense | ✓ | | |
| `robust_learning_rate` | RobustLearningRateDefense | | | |
| `krum` / `multikrum` | KrumDefense | ✓ | | |
| `slsgd` | SLSGDDefense | ✓ | ✓ | |
| `geo_median` | GeometricMedianDefense | | ✓ | |
| `weak_dp` | WeakDPDefense | | | |
| `cclip` | CClipDefense | ✓ | | ✓ |
| `wise_median` | CoordinateWiseMedianDefense | | ✓ | |
| `rfa` | RFADefense | | ✓ | |
| `foolsgold` | FoolsGoldDefense | ✓ | | |
| `3sigma_foolsgold` | ThreeSigmaDefense_Foolsgold | ✓ | | |
| `3sigma_geo` | ThreeSigmaGeoMedianDefense | ✓ | | |
| `3sigma` | ThreeSigmaDefense | ✓ | | |
| `crfl` | CRFLDefense | | | ✓ |
| `trimmed_mean` | CoordinateWiseTrimmedMeanDefense | ✓ | | |
| `anomaly_detection` | OutlierDetection | ✓ | | |

### 2.5 外部注册自定义防御示例

```python
from fedml.core.security.fedml_defender import FedMLDefender, PHASE_BEFORE
from fedml.core.security.defense.defense_base import BaseDefenseMethod

class MyNewDefense(BaseDefenseMethod):
    def __init__(self, config):
        self.threshold = getattr(config, "my_threshold", 0.5)

    def defend_before_aggregation(self, raw_client_grad_list, extra_auxiliary_info=None):
        # 自定义过滤逻辑
        return [g for g in raw_client_grad_list if self._is_clean(g)]

    def _is_clean(self, grad_tuple):
        ...

# 在入口脚本中调用（在 FedMLRunner.run() 之前）
FedMLDefender.register_defense("my_new_defense", MyNewDefense, {PHASE_BEFORE})
```

YAML 配置：
```yaml
enable_defense: true
defense_type: "my_new_defense"
my_threshold: 0.3
```

### 2.6 兼容性

| 项目 | 兼容性 |
|------|--------|
| 所有现有 YAML 配置中的 `defense_type` 值 | ✅ 完全兼容 — 所有 17 个类型字符串保持不变 |
| `ServerAggregator` 基类 | ✅ 无变化 — 仍通过 `FedMLDefender.get_instance()` 调用 |
| `BaselineAggregator` | ✅ 无变化 — Bulyan 手动路径保持不变 |
| `VeriFLv16Aggregator` | ✅ 无影响 — 该聚合器本身重写了 `on_before_aggregation` 和 `aggregate`，不经过 FedMLDefender |

---

## 3. C-2/C-3：VeriFL → v16 重命名

### 3.1 改动原因

后续将引入 VeriFL v18 版本（见 `重生推进方案_实施/VeriFL_v16_to_v18f_升级方案.md`），需要在命名上区分版本。

### 3.2 重命名映射

| 旧名称 | 新名称 | 文件 |
|--------|--------|------|
| `verifl_aggregator.py` | `verifl_v16_aggregator.py` | `trainer/` |
| `verifl_trainer.py` | `verifl_v16_trainer.py` | `trainer/` |
| `VeriFLAggregator` | `VeriFLv16Aggregator` | 类名 |
| `VeriFLTrainer` | `VeriFLv16Trainer` | 类名 |
| `aggregator_type: "verifl"` | `aggregator_type: "verifl_v16"` | 配置值 |
| 日志前缀 `VeriFL ...` | 日志前缀 `VeriFL_v16 ...` | 日志消息 |

### 3.3 受影响的导入

| 文件 | 旧 | 新 |
|------|----|----|
| `trainer/__init__.py` | `from .verifl_trainer import VeriFLTrainer` | `from .verifl_v16_trainer import VeriFLv16Trainer` |
| `main_fedml_shieldfl.py` | `from trainer.verifl_trainer import VeriFLTrainer` | `from trainer.verifl_v16_trainer import VeriFLv16Trainer` |

---

## 4. 冒烟测试

### 4.1 测试矩阵

| 测试 | 类型 | 验证目标 | 结果 |
|------|------|----------|------|
| T1.1 | 单元 | `available_defenses()` 返回 ≥16 个防御 | ✅ PASS |
| T1.2 | 单元 | 未知防御类型抛出 `ValueError` | ✅ PASS |
| T2.1 | 单元 | 所有 17 个内置防御的延迟导入正确解析为 `BaseDefenseMethod` 子类 | ✅ PASS |
| T3.1 | 回归 | 注册表阶段元数据与旧硬编码 if-elif 列表完全一致 | ✅ PASS |
| T4.1 | 集成 | 自定义防御注册 → init() → defend_before_aggregation() 全链路 | ✅ PASS |
| T5.1 | 集成 | 重命名后的模块可 import；旧模块名 import 抛出 `ImportError` | ✅ PASS |
| T6.1 | 回归 | `init()` 在 krum → disabled → foolsgold 切换间完全隔离状态 | ✅ PASS |
| T7.1 | E2E | FedAvg 无防御 3 轮训练（GPU） | ✅ PASS |
| T7.2 | E2E | FedAvg + Krum 防御 3 轮训练（GPU） | ✅ PASS |
| T7.3 | E2E | FedAvg + model_replacement 攻击 3 轮训练（GPU） | ✅ PASS |
| T7.4 | E2E | FedAvg + model_replacement + Krum 3 轮训练（GPU） | ✅ PASS |

### 4.2 运行方法

```bash
# T1-T6（纯 Python，无 GPU）
cd python/examples/federate/prebuilt_jobs/shieldfl
python tests/smoke_test_pluggable_defense.py

# T7（需要 GPU + MPI）
bash tests/smoke_e2e.sh --gpu_id 1
```

---

## 5. 冒烟测试中发现并修复的问题

| 问题 | 文件 | 修复 |
|------|------|------|
| `_normalize_trigger()` 中残留旧类名 `VeriFLTrainer._NORM_PARAMS` | `verifl_v16_trainer.py:77` | → `VeriFLv16Trainer._NORM_PARAMS` |
| T7.3/T7.4 首次失败（wandb/numpy 不兼容） | 运行环境 | 确保 tmux 中激活 `.venv`（系统 Python 的 wandb 与 numpy 2.0 不兼容） |

---

## 6. 已知限制 & 后续事项

| 编号 | 描述 | 优先级 |
|------|------|--------|
| L-1 | `VeriFLv16Aggregator` 仍完全绕过 `FedMLDefender`（设计本意，非缺陷） | 信息 |
| L-2 | Bulyan 仍未注册到 `_DEFENSE_REGISTRY`；`BaselineAggregator` 中手动实例化 | 低 |
| L-3 | `Trimmed Mean` 实现仍为假实现（返回 sample_num） | 中 |
| L-4 | 远程 site-packages 中的 `fedml_defender.py` 需手动覆盖（editable install 未正确生效） | 运维 |
