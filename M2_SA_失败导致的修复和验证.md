# ShieldFL 统一重构方案

> **日期**：2026-04-06  
> **起点**：M2 Scaling Attack 实验失败（模型崩溃）+ 基础设施审计 28 项发现  
> **方法论**：以代码实际行为为准，以原始论文假设为标尺，拒绝一切未经验证的算法性改动

---

## 0. 现状诊断（基于代码审查，非文档推断）

### 0.1 代码实际状态核验

| 项目 | 代码实际状态 | 文档声称 | 差异 |
|------|------------|---------|------|
| LF D1~D7 缺陷 | ✅ 全部已修复（commit 02a90c96） | 已修复 | 一致 |
| F-6 投毒数据旁路 | ❌ **未修复** — `FedMLTrainer.train()` 仍传 `self.train_local`（干净数据），投毒 DataLoader 存入 `self.local_train_dataset` 但从未被读取 | 未修复，列为 W-M0-8 | 一致 |
| VeriFL 覆写管线 | ❌ `on_before_aggregation` + `aggregate` 两个管线方法被完全覆写，FedMLDefender 在 VeriFL 路径下完全失效 | 已确认 | 一致 |
| `aggregator_type` 强制覆写 | ❌ `verifl_aggregator.py` L69: `setattr(args, "aggregator_type", "verifl")` | 已确认 | 一致 |
| VeriflDefense 文件 | ❌ 不存在 | 待创建 | — |
| ShieldFLAggregator 文件 | ❌ 不存在 | 待创建 | — |
| DefenseGPUContext 文件 | ❌ 不存在 | 待创建 | — |
| Trimmed Mean | ❌ 假实现（`compute_a_score` = `return sample_num`） | 不在范围 | 一致 |
| Bulyan | ❌ 未注册到 FedMLDefender | 不在范围 | 一致 |

### 0.2 M2 Scaling Attack 失败的两个根因

**根因 1：K×γ/N 超调**

Bagdasaryan et al. (2020) 的 Model Replacement 攻击假设 **单一恶意客户端**（K=1），缩放系数 γ=N 使得 FedAvg 后全局模型被恶意模型精确替换：

$$G^{(t+1)} = \frac{1}{N}\bigl[\gamma(W_m - G^{(t)}) + G^{(t)} + (N-1)W_b\bigr] \approx W_m \quad (\text{当 } K=1, \gamma=N, W_b \approx G)$$

当前代码设置 K=3, γ=10, N=10，有效放大倍数 Kγ/N = 3.0，导致每轮 3 倍超调，5 轮级联后模型崩溃。

**根因 2：聚合管线不是 FedAvg**

Scaling_实施定稿.md 冻结聚合器为 "FedAvg"，但实际运行路径经过 VeriFL 四阶段（GA 搜索 → L2 投影 → Server Momentum → BN 重校准），完全绕过 `FedMLAggOperator.agg()`。Scaling Attack 的数学推导前提被彻底违反。

### 0.3 关键评估：文档提出了什么，我拒绝了什么

学术方文档（重生推进方案.md）在 M0 架构重构中同时包含了 **VeriFL v16→v18f 算法升级**（相对增量范数正则、增量空间单边裁剪、稀疏探针种群初始化）。我对此持保留意见：

| 变更 | 文档建议 | 本方案决策 | 理由 |
|------|---------|-----------|------|
| VeriFL 迁为 BaseDefenseMethod | ✅ 采纳 | ✅ 采纳 | 架构修正，必须 |
| 移除 BN 重校准 | ✅ 采纳 | ✅ 采纳 | 消除不公平比较、double dipping |
| Momentum 跳过 BN buffers | ✅ 采纳 | ✅ 采纳 | 数学范畴错误修正 |
| **Phase 1: 绝对范数→相对增量范数** | ✅ 采纳 | ⛔ **推迟** | **算法级变更**，未经实验验证的新正则项不应与架构修复同时引入 |
| **Phase 2: 双边投影→单边裁剪** | ✅ 采纳 | ⛔ **推迟** | 同上。改变裁剪语义可能引入新的不可预测行为 |
| **λ 从 0.1 → 32.0** | ✅ 采纳 | ⛔ **推迟** | 量纲变更必须在新正则项启用后才有意义 |
| **稀疏探针种群初始化** | ✅ 采纳 | ⛔ **推迟** | GA 搜索策略变更，应在基线稳定后独立评估 |
| DefenseGPUContext 共享重构 | ✅ 采纳 | ⚠️ **简化** | 先做最小迁移（GPUAccelerator → VeriflDefense 内部使用），不预设 FLTrust 接口需求 |
| K=1 修正 | ✅ 采纳 | ✅ 采纳 | 恢复 Bagdasaryan 原文语义，仅配置变更 |
| F-6 投毒路径修复 | ✅ 采纳 | ✅ 采纳 | 阻塞性 bug |

**核心原则：新引入的算法改动不得与架构修复同批上线。** 理由：

1. v18f 的相对增量范数、单边裁剪是**未在当前实验载体上验证过的新算法**，其效果未知
2. 如果同时修改架构 + 算法，且后续实验出现异常，无法区分是架构迁移引入的 bug 还是新算法本身的问题
3. 用户明确要求"不要再出现新引入的算法干扰标准算法的问题"——v18f 的改动恰好属于此类风险
4. 正确的做法是：先用 **v16 原始算法**完成架构迁移并验证等价性，再作为独立阶段引入 v18f

---

## 1. 分阶段实施总览

```
Phase 0: 基线安全隔离              ← 最高优先级，0 新算法引入
  ├─ P0.1: 创建 ShieldFLAggregator（不覆写任何管线方法）
  ├─ P0.2: 简化 main 为单一 aggregator
  ├─ P0.3: 修复 F-6（投毒数据旁路）
  ├─ P0.4: run_experiment.sh 适配
  └─ P0.5: 基线回归验证（defense_type=none 的 FedAvg 结果与 M1.5 一致）
      │
      ▼ 基线验证通过（FedAvg 行为完全等价于 M1.5）
      │
Phase 1: 攻击实验重跑              ← 在已验证的纯 FedAvg 上
  ├─ P1.1: LF 实验（F-6 修复后首次真实投毒）
  ├─ P1.2: Scaling 实验（K=1, γ=10, defense_type=none）
  └─ P1.3: γ=1 控制实验
      │
      ▼ 攻击基准数据拿到
      │
Phase 2: VeriFL 防御迁移            ← 保持 v16 算法不变，仅架构迁移
  ├─ P2.1: 创建 VeriflDefense(BaseDefenseMethod)，v16 算法原样迁移
  ├─ P2.2: 注册到 FedMLDefender
  ├─ P2.3: 迁移 GPUAccelerator（最小化改动）
  ├─ P2.4: 等价性验证（defense_type=verifl 与旧 VeriFL 路径数值一致）
  └─ P2.5: 标记旧文件废弃
      │
      ▼ 架构迁移完成且等价性已验证
      │
Phase 3: v18f 算法升级（可选，独立评估）
  ├─ P3.1: Phase 1 适应度函数改为相对增量范数
  ├─ P3.2: Phase 2 改为增量空间单边裁剪
  ├─ P3.3: 独立验证 v18f vs v16 的防御效果差异
  └─ P3.4: 确认 v18f 不劣化 clean baseline
```

**关键约束**：每个 Phase 必须通过门禁验证后才能进入下一个 Phase。任何门禁失败必须在当前 Phase 内解决，不得携带到下一 Phase。

---

## 2. Phase 0：基线安全隔离（详细变更）

### P0.1 创建 ShieldFLAggregator

**新建文件**：`python/examples/federate/prebuilt_jobs/shieldfl/trainer/shieldfl_aggregator.py`

```python
class ShieldFLAggregator(ServerAggregator):
    """
    ShieldFL 唯一聚合器。
    仅实现抽象方法，不覆写任何管线方法。
    defense_type 通过 FedMLDefender 控制聚合行为。
    """
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters, strict=True)

    def test(self, test_data, device, args):
        # 从 verifl_aggregator.py / baseline_aggregator.py 迁移
        # 包含 clean accuracy 评估 + ASR 评估（若 eval_asr=true）
        ...

    def test_all(self, ...):
        ...
```

**禁止包含**：
- ❌ `def on_before_aggregation` — 不覆写
- ❌ `def aggregate` — 不覆写
- ❌ `def on_after_aggregation` — 不覆写
- ❌ 任何 GPUAccelerator / MicroGABase 引用
- ❌ 任何 `setattr(args, "aggregator_type", ...)` 操作

**效果**：当 `defense_type=none` 时，基类 `ServerAggregator.aggregate()` 自动调用 `FedMLAggOperator.agg()`（纯 FedAvg）。当 `defense_type=verifl`（Phase 2 后），基类自动调度到 `FedMLDefender.defend_on_aggregation()`。

### P0.2 简化 main_fedml_shieldfl.py

**当前代码**：
```python
aggregator_type = str(getattr(args, "aggregator_type", "verifl")).strip().lower()
if aggregator_type == "fedavg":
    aggregator = BaselineAggregator(...)
else:
    aggregator = VeriFLAggregator(...)
```

**修改后**：
```python
aggregator = ShieldFLAggregator(model, args)
```

移除所有 `aggregator_type` 分支逻辑和相关 import。

### P0.3 修复 F-6（投毒数据旁路）

**修改文件**：verifl_trainer.py

**在 `train(self, train_data, device, args)` 方法开头添加**：

```python
def train(self, train_data, device, args):
    # F-6 fix: FedMLTrainer.train() passes self.train_local (clean data),
    # but ClientTrainer.update_dataset() stores poisoned data in self.local_train_dataset.
    # Redirect to poisoned DataLoader when data poisoning attack is active.
    if hasattr(self, 'local_train_dataset') and self.local_train_dataset is not None:
        train_data = self.local_train_dataset

    # ... rest of training unchanged
```

**根因**：`FedMLTrainer.train()` (fedml_trainer.py) 调用 `self.trainer.train(self.train_local, ...)` 传入干净数据。`ClientTrainer.update_dataset()` 将投毒后的 DataLoader 存入 `self.local_train_dataset`——一个 `train()` 方法从未读取的变量。`poison_data()` 返回全新 DataLoader 对象（非原地修改），因此原始 `self.train_local` 完全不受影响。此修复确保投毒数据在被记录后也能被实际训练使用。

**影响范围**：仅 Label Flipping（唯一使用 DATA 路径的攻击）。Scaling Attack 不受影响（使用 MODEL 路径 + trainer 内联后门注入）。

### P0.4 更新 run_experiment.sh

**修改文件**：run_experiment.sh

核心变更：
1. 移除 `--aggregator` 参数及其处理逻辑
2. `--defense none` 时生成：
```yaml
enable_defense: false
defense_type: "none"
```
3. `--defense verifl` 时生成（Phase 2 后生效）：
```yaml
enable_defense: true
defense_type: "verifl"
```
4. 当 `--attack model_replacement` 且 PMR=0.1 时自动导出 `byzantine_client_num: 1`

### P0.5 基线回归验证（门禁）

**验证目标**：`defense_type=none` 时的聚合行为与 M1.5 冻结结果数值等价。

**验证方法**：运行 1 组 smoke test（CIFAR-10, α=0.5, seed=0, 5 轮, 无攻击, `defense_type=none`）。

**通过条件**：
| 编号 | 条件 | 验证方式 |
|------|------|---------|
| AC-P0-1 | exit code = 0，生成 metrics 文件 | 脚本输出 |
| AC-P0-2 | 日志中 `FedMLAggOperator.agg()` 被调用 | grep 日志 |
| AC-P0-3 | 日志中**不出现**任何 GA/L2 投影/momentum/BN 重校准字样 | grep 日志 |
| AC-P0-4 | `shieldfl_aggregator.py` 中不存在 `def on_before_aggregation` / `def aggregate` / `def on_after_aggregation` | grep 代码 |
| AC-P0-5 | 全项目 grep 不出 `VeriFLAggregator` 或 `BaselineAggregator` 的活跃引用 | grep 代码 |
| AC-P0-6 | F-6 修复验证：`attack_type=label_flipping` 的 5 轮 smoke test 中，恶意客户端实际使用投毒数据（日志中标签分布与干净时不同） | 日志对比 |

**⚠️ 失败处理**：如果 AC-P0-1~AC-P0-5 任一失败，说明架构迁移引入了新 bug。必须在 Phase 0 内修复。绝不进入 Phase 1。

---

## 3. Phase 1：攻击实验重跑（详细配置）

### P1.1 Label Flipping 实验

**前置条件**：Phase 0 全部门禁通过

**实验配置**（继承 LF_实施规格.md，仅聚合路径变更为真正的 FedAvg）：

| 参数 | 值 |
|------|-----|
| attack_type | label_flipping |
| defense_type | **none**（纯 FedAvg） |
| PMR | 0.3 |
| 映射 | [0..9] → [9..0] |
| eval_asr | false |
| 实验矩阵 | 2 数据集 × 4 α × 3 seed = 24 组 |

**注意**：这是 F-6 修复后 LF 攻击**首次产出真实投毒效果**的实验。不与 M2 旧数据对比（旧数据因 F-6 完全无效），改为与 M1.5 clean baseline 对比。

**验收标准**：
| 编号 | 条件 |
|------|------|
| AC-P1-1 | 24 组全部运行完成，exit code = 0 |
| AC-P1-2 | LF 实验的 MTA ≤ M1.5 同配置 clean baseline（攻击不应提升精度） |
| AC-P1-3 | 至少 1 个 α 配置的 Mean MTA Drop ≥ 1.0 pp（攻击起作用）|
| AC-P1-4 | 若所有 α 配置 Drop < 1.0 pp → 切 α=0.1 验证是否因 F-6 修复确实生效但效果被稀释 |

### P1.2 Scaling Attack 实验

**关键修订**（基于 Bagdasaryan 原文假设）：

| 参数 | M2 旧值 | 修订值 | 理由 |
|------|---------|-------|------|
| byzantine_client_num (K) | 3 | **1** | 恢复单攻击者语义：Kγ/N = 1×10/10 = 1.0（精确 model replacement） |
| malicious_client_ids | [0,1,2] | **[0]** | K=1 的直接结果 |
| PMR | 0.3 | **0.1** | K/N 比例 |
| defense_type | verifl (隐式) | **none** | 纯 FedAvg，消除 VeriFL 管线干扰 |

**不变参数**：γ=10, target_label=0, trigger_size=3, trigger_value=1.0, backdoor_per_batch=20, 末段 5 轮攻击窗口

**代码变更需求**：**零代码变更**。全部通过配置控制：
- `byzantine_client_num: 1` → 现有代码 `model_replacement_backdoor_attack.py` 中 `range(self.byzantine_client_num)` 自动变为 `[0]`
- `defense_type: none` → 基类管线自动走 FedAvg

**数学验证（K=1, γ=10, N=10）**：

$$G^{(t+1)} = \frac{1}{10}\bigl[10(W_m - G) + G + 9G\bigr] = W_m \quad (\text{精确 model replacement})$$

**实验矩阵**：2 数据集 × 4 α × 3 seed = 24 组 + 3 组 γ=1 控制实验

**验收标准**：
| 编号 | 条件 |
|------|------|
| AC-P1-5 | 27 组全部运行完成 |
| AC-P1-6 | CIFAR-10 至少 3 个 α 的 mean ASR ≥ 0.80 |
| AC-P1-7 | CIFAR-10 γ=10 的 clean accuracy **不崩溃**（> 0.50，非随机猜测 0.10）|
| AC-P1-8 | ΔASR = mean(ASR, γ=10) − mean(ASR, γ=1) ≥ 0.30（缩放因果性） |
| AC-P1-9 | MNIST 至少 3 个 α 的 mean ASR ≥ 0.90 |

**AC-P1-7 是新增的关键检查**：M2 时所有 CIFAR-10 γ=10 实验 accuracy=0.10（崩溃）。K=1 修复后 clean accuracy 应保持正常，因为恶意模型 $W_m$ 本身经过了正常训练（仅 20/64 batch 被后门替换）。

**AC-P1-8 因果性预期**：K=1, γ=1 时恶意贡献仅占 10%（vs M2 旧实验的 30%），后门被 9 个良性更新大幅稀释，预期 γ=1 ASR 显著低于 γ=10，ΔASR 有充分空间。

---

## 4. Phase 2：VeriFL 防御迁移（详细变更）

### 核心原则：**v16 算法原样迁移，不做任何算法性改动（除 BN 修复外）**

### P2.1 创建 VeriflDefense

**新建文件**：`python/fedml/core/security/defense/verifl_defense.py`

```python
class VeriflDefense(BaseDefenseMethod):
    def __init__(self, config):
        # 从 config 读取 GA 参数、momentum 参数
        # **保持 v16 默认值不变**：lambda_reg=0.1, pop_size=15, generations=10

    def defend_on_aggregation(
        self,
        raw_client_grad_list,
        base_aggregation_func=None,
        extra_auxiliary_info=None
    ) -> OrderedDict:
        # 从 verifl_aggregator.py 的 aggregate() 迁移，保持算法完全等价
        # Phase 1: GA 搜索（v16 绝对模型范数正则，λ=0.1）
        # Phase 2: L2 投影（v16 双边投影）
        # Phase 3: Server Momentum（**修复**：BN buffers 跳过 momentum）
        # Phase 4: **移除** BN 重校准
```

**算法变更（仅限 bug 修复，非功能性改动）**：

| 项目 | v16 当前行为 | 迁移后行为 | 性质 |
|------|------------|-----------|------|
| Phase 3 Momentum 范围 | 全参数（含 BN buffers） | trainable\_mask=True 的参数 | **bug 修复**：对 BN 统计估计量施加 SGD 动量是范畴错误 |
| Phase 4 BN 重校准 | 在验证集上 forward pass 刷新 | **移除** | **消除 double dipping + 不公平比较 + 全局 RNG 重置** |
| BN buffers 聚合 | GA α\*-加权（但被 recal 覆写） | GA α\*-加权（最终生效值） | recal 移除后的自然结果 |
| `sample_num` | 被 `_` 丢弃 | **保留并传递** | 修复 F-5（VeriFL 丢弃样本量信息） |

**不变动**（v16 保持原样）：
- ✅ GA 适应度函数：`cost = loss + λ × model_norm`，`λ=0.1`
- ✅ L2 投影：双边投影到 anchor 范数
- ✅ 种群初始化：1 个 FedAvg + 随机补齐，无稀疏探针
- ✅ 遗传操作：锦标赛选择、线性交叉、高斯变异

### P2.2 注册到 FedMLDefender

**修改文件**：fedml_defender.py

1. 顶部新增常量：`DEFENSE_VERIFL = "verifl"`
2. `init()` 中新增 elif 分支
3. `defend_on_aggregation` 的 defense_type 集合中添加 `DEFENSE_VERIFL`

### P2.3 迁移 GPUAccelerator

**策略**：最小化改动。GPUAccelerator 当前与 VeriFLAggregator 耦合（通过 MicroGABase 多继承）。迁移时：

1. 将 `gpu_accelerator.py` 中的 `GPUAccelerator` 类保持大部分不变
2. 移除 `recalibrate_batchnorm()` 方法
3. VeriflDefense 在 `__init__` 中实例化 `GPUAccelerator`
4. 将 MicroGABase 的遗传操作逻辑直接迁入 VeriflDefense 或作为其内部辅助类

**不重构为 DefenseGPUContext**：这是文档提出的前瞻性重构（为 FLTrust 等未来防御预留接口）。当前阶段不需要——FLTrust 不在此方案范围内。当实际需要时再重构，避免过度工程。

### P2.4 等价性验证（门禁）

**验证目标**：`defense_type=verifl` 的新路径与旧 VeriFLAggregator 路径在除 BN 修复外的行为完全等价。

**验证方法**：

| 编号 | 条件 |
|------|------|
| AC-P2-1 | smoke test（5 轮，`defense_type=verifl`）exit code = 0 |
| AC-P2-2 | 日志中出现 GA 搜索、L2 投影、momentum 更新 |
| AC-P2-3 | 日志中**不出现** BN 重校准 |
| AC-P2-4 | 基类三阶段管线完整执行（`on_before_aggregation` → `aggregate` → `on_after_aggregation`） |
| AC-P2-5 | `defense_type=none` 的 smoke test 不受 VeriFL 注册影响（结果与 P0.5 门禁一致）|
| AC-P2-6 | momentum 不作用于 BN buffers：构造含 BN 模型的单元测试验证 `running_mean/running_var` 不受 momentum 影响 |

**AC-P2-5 极端重要**：这是确保 VeriFL 防御注册不干扰无防御基线的关键检查。如果此条失败，说明 FedMLDefender 的状态管理存在泄漏。

### P2.5 标记旧文件废弃

- `verifl_aggregator.py` → `.deprecated`
- `baseline_aggregator.py` → `.deprecated`
- 旧 `gpu_accelerator.py` → `.deprecated`（如果已被新实现替代）

---

## 5. Phase 3：v18f 算法升级（可选，需独立评审）

⚠️ **此阶段不属于本方案的核心范围。仅在 Phase 2 完成并经回归验证后，作为独立的算法改进评估。**

如果决定执行，变更内容为：

| 项目 | v16 → v18f |
|------|-----------|
| Phase 1 适应度 | `cost = loss + 0.1 × model_norm` → `cost = loss + 32.0 × delta_norm / (global_prev_norm + ε)` |
| Phase 2 投影 | 双边投影（scale 可 >1） → 增量空间单边裁剪（scale ≤ 1） |
| 种群初始化 | FedAvg + 随机 → FedAvg + 稀疏探针 + 随机 |
| GPU 接口 | `(loss, model_norm)` → `(loss, model_norm, delta_norm)` |

**独立验收要求**：
1. v18f 在 clean baseline（无攻击）上不劣化 FedAvg 结果
2. v18f 在 LF + Scaling 攻击下防御效果不差于 v16
3. v18f 的超参数（λ\_f=32.0）经过 sensitivity 分析

---

## 6. 全局冻结清单

以下组件在整个重构过程中**不可改动**：

### 6.1 实验载体冻结（继承 M1.5）

| 维度 | 冻结值 |
|------|-------|
| 训练模式 | cross-silo / MPI |
| 客户端总数 / 每轮参与 | 10 / 10 |
| CIFAR-10 | ResNet18, 100 轮 |
| MNIST | LeNet5, 50 轮 |
| local epochs | 1 |
| batch\_size | 64 |
| learning\_rate | 0.01 |
| weight\_decay | CIFAR-10=1e-4, MNIST=0 |
| client momentum | 0.9 |
| α 网格 | {0.1, 0.3, 0.5, 100} |
| seed 网格 | {0, 1, 2} |

### 6.2 代码冻结（不可改动的组件）

| 组件 | 文件 | 理由 |
|------|------|------|
| LF 攻击逻辑 | `label_flipping_attack.py` | D1~D7 已修复验证 |
| Scaling 攻击逻辑 | `model_replacement_backdoor_attack.py` | 代码正确，仅配置变更 |
| Scaling 客户端训练 | `verifl_trainer.py`（除 F-6 外） | 后门注入逻辑已正确 |
| ASR 评估 | `eval/asr.py` | 已验证 |
| FedAvg 聚合算子 | `agg_operator.py` | 数学正确 |
| 数据加载 | `data/data_loader.py` | non-IID 划分已隔离 |
| 基类管线 | `server_aggregator.py` | FedML 框架核心 |

---

## 7. 文件变更总图

```
Phase 0 新建（1 个）
└── python/examples/.../shieldfl/trainer/shieldfl_aggregator.py     ← P0.1

Phase 0 修改（4 个）
├── python/examples/.../shieldfl/main_fedml_shieldfl.py             ← P0.2（简化为单一 aggregator）
├── python/examples/.../shieldfl/trainer/__init__.py                ← P0.2 补遗（re-export ShieldFLAggregator，见 E-12）
├── python/examples/.../shieldfl/trainer/verifl_trainer.py          ← P0.3（F-6 修复）
└── python/examples/.../shieldfl/scripts/run_experiment.sh          ← P0.4（移除 aggregator 参数）

Phase 1 修改（1 个）
└── python/examples/.../shieldfl/scripts/run_experiment.sh          ← PMR=0.1/K=1 配置

Phase 2 新建（1 个）
└── python/fedml/core/security/defense/verifl_defense.py            ← P2.1

Phase 2 修改（1 个）
└── python/fedml/core/security/fedml_defender.py                    ← P2.2（新增 elif）

Phase 2 废弃（2~3 个）
├── python/examples/.../shieldfl/trainer/verifl_aggregator.py       → .deprecated
├── python/examples/.../shieldfl/trainer/baseline_aggregator.py     → .deprecated
└── python/examples/.../shieldfl/trainer/gpu_accelerator.py         → .deprecated（可选）

冻结不变
├── python/fedml/core/security/attack/model_replacement_backdoor_attack.py
├── python/fedml/core/security/attack/label_flipping_attack.py
├── python/fedml/core/security/common/utils.py
├── python/fedml/ml/aggregator/agg_operator.py
└── python/fedml/core/alg_frame/server_aggregator.py
```

---

## 8. 审计发现处置映射

| 编号 | 严重级 | Phase | 处置 |
|------|-------|-------|------|
| F-1 | 🔴 Fatal | P0 + P2 | P0: ShieldFLAggregator 不覆写管线 → defense\_type=none 时 FedMLDefender 正常调度; P2: VeriflDefense 注册后 defense\_type=verifl 也走正常管线 |
| F-2 | 🔴 Fatal | 不在范围 | Trimmed Mean 假实现，不影响当前实验（使用 defense\_type=none 或 verifl） |
| F-3 | 🔴 Fatal | 不在范围 | Bulyan 未注册，同上 |
| F-4 | 🔴 Fatal | P0 | `defense_type=none` → 基类 `aggregate()` → `FedMLAggOperator.agg()` |
| F-5 | 🔴 Fatal | P2 | VeriflDefense 不再丢弃 `sample_num` |
| F-6 | 🔴 Fatal | P0 | `verifl_trainer.py` train() 重定向到投毒 DataLoader |
| S-1 | 🟠 Severe | P2 | Momentum 仅作用于 trainable\_mask=True |
| S-2 | 🟠 Severe | P0 | 基类管线不被覆写 → DP clip 正常执行 |
| S-3 | 🟠 Severe | P2 | 移除 BN 重校准，所有路径统一 |
| S-4 | 🟠 Severe | 文档记录 | L2 投影不保护 BN buffers — 设计决策接受 |
| S-5 | 🟠 Severe | P2 | 部分修复（移除 recal 后 GA-聚合差异缩小） |
| M-2 | 🟡 Mod | P0 | 旧 aggregator 不再使用，`setattr` 随之消失 |
| M-4 | 🟡 Mod | P0 | `sort_client_updates` 随旧 aggregator 消失 |
| M-5 | 🟡 Mod | P2 | `recalibrate_batchnorm` 移除，不再重置全局 RNG |
| M-6 | 🟡 Mod | P2 | BN 重校准移除 |
| M-8 | 🟡 Mod | P2 | BN recal 移除后不再有叠加问题 |
| 其余 | 🟡🔵⚪ | 不在范围 | 不影响当前实验 |

---

## 9. 风险清单

| # | 风险 | 可能性 | 应对 |
|---|------|-------|------|
| R-1 | Phase 0 后 FedAvg 结果与 M1.5 不一致 | 低 | `BaselineAggregator.aggregate()` 在 `defense_type=none` 时已走 `super().aggregate()` → `FedMLAggOperator.agg()`。新路径等价。若不一致，检查 `test()` 方法迁移是否丢失指标计算逻辑 |
| R-2 | F-6 修复后 LF 效果仍不显著 | 中 | α=0.5 + PMR=0.3 下良性数据稀释预期。先确认投毒日志正确（标签分布变化），再看 α=0.1 效果。若全部 α drop < 1pp，需排查 DataLoader 替换是否真正生效 |
| R-3 | K=1 Scaling ASR 不达标 | 低 | K=1,γ=10 在纯 FedAvg 上是精确 model replacement，理论上 ASR 应很高。若不达标：(a) 检查 FedAvg 加权是否真的是等权（样本量加权 vs 均匀加权可能有差异）；(b) 检查 5 轮窗口是否足够 |
| R-4 | Phase 2 VeriFL defense 迁移后行为不等价 | 中 | 保留旧 aggregator 作为 A/B 对比。用相同输入对比新旧 GA 输出。BN 修复会导致数值差异——这是预期的，需记录 |
| R-5 | v18f 算法升级引入新问题 | — | 不在本方案核心范围。若执行，作为独立 Phase 有独立门禁 |

---

## 10. 本方案与文档方案的关键差异总结

| 维度 | 文档方案（重生推进方案.md） | 本方案 | 理由 |
|------|-------------------------|-------|------|
| 分阶段数 | 3（M0 + M1a + M1b） | 4（P0 + P1 + P2 + P3） | 将"架构修复"和"VeriFL 迁移"拆开，在纯 FedAvg 验证后再引入防御 |
| v18f 升级 | M0 内完成 | P3（独立可选） | 避免算法+架构同时变更导致的不可追溯 |
| DefenseGPUContext | M0 内创建（为 FLTrust 预留） | 不创建 | 过度工程，当前无消费者 |
| VeriFL 迁移时机 | M0（与基线修复同批） | P2（攻击实验完成后） | 确保纯 FedAvg 基线先固化，再引入防御层 |
| 基线验证 | AC-M0-1 (5 轮 smoke) | AC-P0-1~6 + Phase 1 全量实验 | 更严格——不仅 smoke，还要全部攻击实验在纯 FedAvg 上跑通 |
| Phase 1 vs Phase 2 先后 | M1a/M1b 在 M0 后并行 | Phase 1（纯 FedAvg 实验）先于 Phase 2（VeriFL 迁移） | 攻击实验在纯 FedAvg 上的结果是后续对比 VeriFL 防御效果的**不可或缺的基准** |

---

## 11. 代码审查勘误（基于 2026-04-06 代码库全量核验）

> 以下问题由逐文件审计发现，与文档规划存在偏差或遗漏。每条标注严重等级和影响阶段。

### E-1 🔴 FedAvg 加权方式误述——数学推导前提需修正

**文档声称**（§0.2）：

$$G^{(t+1)} = \frac{1}{N}\bigl[\gamma(W_m - G^{(t)}) + G^{(t)} + (N-1)W_b\bigr] \approx W_m$$

此公式隐含 **等权 FedAvg**（每客户端权重 $1/N$）。

**代码实际行为**（`agg_operator.py` L39–45 `torch_aggregator`）：

```python
w = local_sample_number / training_num   # ← 样本数加权，非 1/N
```

精确公式为：

$$G^{(t+1)} = \sum_i \frac{n_i}{\sum_j n_j} W_i'$$

精确 model replacement 需要 $\gamma = \frac{\sum_j n_j}{n_m}$，而非 $\gamma = N$。

**为何当前设置下碰巧等价**：`data_loader.py` 中 `max_samples_per_client=300` 对所有客户端做了硬截断。由于 Dirichlet 分配在 48000 样本（CIFAR-10 训练集 50000 − val 500 − trust 500）/ 10 客户端下，每客户端期望分配约 4800 样本，远超 300 上限。因此所有客户端均被截断至 $n_i = 300$，权重退化为 $300/3000 = 1/10 = 1/N$。

**影响**：当前 K=1, γ=10, N=10 配置在实践中能实现精确 model replacement。但任何改变 `max_samples_per_client` 或使用不同数据集的实验都可能打破此前提。

**建议修正**：在 §0.2 数学验证中显式注明 "在 `max_samples_per_client=300` 且所有客户端截断至相同样本数的前提下，样本数加权 FedAvg 等价于等权 FedAvg"。在 §6.1 冻结清单中将 `max_samples_per_client: 300` 列为冻结参数。

---

### E-2 🔴 `server_momentum=0.0` 导致 VeriFL Phase 3 为空操作

**文档声称**（§4 P2.1 / §2 P0.1）：

> Phase 3: Server Momentum（**修复**：BN buffers 跳过 momentum）

暗示 Phase 3 在当前实验中有实际效果。

**代码实际行为**：`run_experiment.sh` L248 生成配置 `server_momentum: 0.0`，`server_lr: 1.0`。

```yaml
server_momentum: 0.0    # ← 全局默认
server_lr: 1.0
```

在 `verifl_aggregator.py` Phase 3 代码中：

```python
velocity = 0.0 * old_velocity + delta       # = delta
updated = old_global + 1.0 * delta           # = ga_aggregated_params
```

**结论**：Phase 3 是数学恒等变换——`final_params = ga_aggregated_params`。**Server Momentum 在所有历史实验中从未实际生效**。

**影响**：
1. S-1 发现（momentum 作用于 BN buffers）在当前配置下从未实际触发
2. Phase 2 中 "修复 BN momentum" 的改动不会改变任何数值结果（与旧路径完全等价）
3. Phase 2 等价性验证（AC-P2-6 BN momentum 单元测试）可以执行但不会暴露真实差异
4. 仅当未来启用非零 `server_momentum` 时此修复才有实际意义

**建议修正**：在 §4 P2.1 中注明 "当前配置 `server_momentum=0.0` 使 Phase 3 为恒等变换。BN momentum 修复为预防性修正，在 `server_momentum > 0` 时才有数值差异。" 同时在 §6.1 冻结清单中添加 `server_momentum: 0.0` 和 `server_lr: 1.0`。

---

### E-3 🟠 `lambda_reg` 实际运行值 ≠ 文档声称的 v16 默认值

**文档声称**（§4 P2.1）：

> **保持 v16 默认值不变**：lambda_reg=0.1

**代码实际行为**：
- `MicroGABase.__init__` 代码默认值：`lambda_reg=0.1`
- `run_experiment.sh` L251 生成配置：`lambda_reg: 0.01`

`VeriFLAggregator.__init__` 通过 `getattr(args, "lambda_reg", 0.1)` 读取，**config 值 0.01 覆盖代码默认值 0.1**。

**影响**：所有历史 VeriFL 实验实际使用 `lambda_reg=0.01`，而非 0.1。Phase 2 VeriflDefense 若初始化为 0.1 将**不等价于旧路径**，AC-P2-4 等价性验证将失败。

**建议修正**：§4 P2.1 中修改为 "从 config 读取 lambda_reg（当前配置值 0.01），不使用硬编码默认值"。

---

### E-4 🟠 `BaselineAggregator` 文档注释与代码行为不符

**`baseline_aggregator.py` 文件头注释声称**：

> 不覆盖 on_before_aggregation / aggregate / on_after_aggregation

**代码实际行为**（`baseline_aggregator.py` L52–60）：

```python
def aggregate(self, raw_client_model_or_grad_list):
    defense_type = str(getattr(self.args, "defense_type", "none")).strip().lower()
    t0 = time.time()
    if defense_type == "bulyan":
        result = self._aggregate_bulyan(raw_client_model_or_grad_list)
    else:
        result = super().aggregate(raw_client_model_or_grad_list)
    ...
```

**`aggregate()` 被覆写**，虽然非 Bulyan 分支委托给 `super()`，但仍是方法覆写。

**影响**：§0.1 代码状态表中仅列出 VeriFLAggregator 的管线覆写问题。`BaselineAggregator` 的 `aggregate()` 覆写未被提及。虽然功能上不影响 `defense_type=none` 的正确性（`super()` 最终调用 `FedMLAggOperator.agg()`），但与代码文档的自描述矛盾——说明文档对旧代码的审计存在遗漏。

**建议修正**：§0.1 表中新增一行："BaselineAggregator aggregate() 覆写 | 覆写但委托 super() | 文档声称不覆写 | 不一致"。

---

### E-5 🟠 ShieldFLAggregator 构造函数签名缺少关键参数

**文档 §2 P0.1 创建 ShieldFLAggregator**：

```python
class ShieldFLAggregator(ServerAggregator):
    def get_model_params(self): ...
    def set_model_params(self, model_parameters): ...
    def test(self, test_data, device, args): ...
```

**文档 §2 P0.2 简化 main**：

```python
aggregator = ShieldFLAggregator(model, args)
```

**问题**：现有两个 aggregator 的 `test()` 方法均依赖 `self.data_assets`（ASR 评估需要 `data_assets.test_loader`）和 `self.device`。`MetricsCollector` 也在 `__init__` 中初始化。仅传 `(model, args)` 将导致：
1. ASR 评估不可用（无 `data_assets`）
2. 设备信息缺失
3. MetricsCollector 虽不依赖 data_assets 但需要启动初始化

**建议修正**：P0.1 构造函数应为：

```python
class ShieldFLAggregator(ServerAggregator):
    def __init__(self, model, args, data_assets=None, device=None):
        super().__init__(model, args)
        self.data_assets = data_assets
        self.device = device or torch.device("cpu")
        metrics_dir = str(getattr(args, "metrics_output_dir", "./results"))
        self._metrics_collector = MetricsCollector(metrics_dir, args)
        self._last_agg_time = None
```

P0.2 对应修改为：

```python
aggregator = ShieldFLAggregator(model=model, args=args, data_assets=data_assets, device=device)
```

---

### E-6 🟡 AC-P0-6 标签分布日志验证不可行

**文档要求**（AC-P0-6）：

> "恶意客户端实际使用投毒数据（日志中标签分布与干净时不同）"

**代码实际状态**：`verifl_trainer.py` 的 `train()` 方法中没有任何标签分布日志输出。训练循环仅记录 loss 值。Label Flipping 的投毒发生在 `ClientTrainer.update_dataset()` 内（由 `LabelFlippingAttack.poison_data()` 重建 DataLoader），重建后的 DataLoader 不包含标签分布统计。

**影响**：AC-P0-6 按当前代码无法直接验证。

**建议修正**：F-6 修复代码中增加一行诊断日志：

```python
def train(self, train_data, device, args):
    if hasattr(self, 'local_train_dataset') and self.local_train_dataset is not None:
        train_data = self.local_train_dataset
    # F-6 diagnostic: log label distribution for this client
    if hasattr(self, 'id') and logging.getLogger().isEnabledFor(logging.INFO):
        label_counts = {}
        for _, labels in train_data:
            for l in labels.tolist():
                label_counts[l] = label_counts.get(l, 0) + 1
        logging.info("Client %s train label_distribution=%s", self.id, dict(sorted(label_counts.items())))
```

或将 AC-P0-6 改为基于训练损失的间接验证："恶意客户端的首轮 loss 显著高于干净基线（因标签翻转导致标签错误率 100%）"。

---

### E-7 🟡 MetricsCollector 文件名中 `aggregator_type` 默认值问题

**当前行为**：`MetricsCollector.__init__` 中 `aggregator_type = str(getattr(args, "aggregator_type", "verifl"))`。两个旧 aggregator 都在 `__init__` 中做 `setattr(args, "aggregator_type", ...)` 来控制此值。

**P0.1 规约**："禁止包含 `setattr(args, 'aggregator_type', ...)`"。

**P0.4 规约**："移除 `--aggregator` 参数"。

**问题**：若 `run_experiment.sh` 不再生成 `aggregator_type` 字段，且 ShieldFLAggregator 不设置它，则 `getattr(args, "aggregator_type", "verifl")` 将回退到默认值 `"verifl"`。指标文件名将包含 `verifl` 字样——即使实际使用的是 FedAvg 路径。

**建议修正**：
1. `run_experiment.sh` 中保留一行固定输出 `aggregator_type: "shieldfl"`（不从命令行参数获取）
2. 或在 P0.1 ShieldFLAggregator 中用 `"shieldfl"` 初始化 MetricsCollector：

```python
if not hasattr(args, 'aggregator_type'):
    setattr(args, 'aggregator_type', 'shieldfl')
```

---

### E-8 🟡 `BaselineAggregator.__init__` 也执行 `setattr(args, "aggregator_type", "fedavg")`

**文档 §0.1 表中仅列出**："`aggregator_type` 强制覆写 | `verifl_aggregator.py` L69"

**遗漏**：`baseline_aggregator.py` L42 也执行：

```python
setattr(args, "aggregator_type", "fedavg")
```

**影响**：当前 `aggregator_type=fedavg` 时此 setattr 是冗余的但无害。需确认新 ShieldFLAggregator 确实不包含任何 setattr，且 §0.1 代码状态表补充此发现。

---

### E-9 🔵 §6.1 冻结清单缺少关键参数

以下参数在实验中对数值结果有实质性影响，但未列入冻结清单：

| 参数 | 当前 config 值 | 位置 | 理由 |
|------|-------------|------|------|
| `max_samples_per_client` | 300 | `run_experiment.sh` | 决定 FedAvg 加权是否退化为等权（E-1）|
| `val_per_class` | 50 | `run_experiment.sh` | 影响 VeriFL fitness 评估质量 |
| `test_subset_size` | 500 | `run_experiment.sh` | 影响 ACC/ASR 评估方差 |
| `server_momentum` | 0.0 | `run_experiment.sh` | 决定 VeriFL Phase 3 是否生效（E-2）|
| `server_lr` | 1.0 | `run_experiment.sh` | 同上 |
| `lambda_reg` | 0.01 | `run_experiment.sh` | VeriFL fitness 正则项系数（E-3）|

**建议修正**：将以上参数加入 §6.1 冻结清单。

---

### E-10 🔵 F-6 修复说明中 "仅 LF 受影响" 需细化

**文档声称**（§2 P0.3）：

> 影响范围：仅 Label Flipping（唯一使用 DATA 路径的攻击）。Scaling Attack 不受影响（使用 MODEL 路径 + trainer 内联后门注入）。

**代码补充证据**：这是正确的。具体路径分析：
- **LF**：`FedMLAttacker.is_data_poisoning_attack()` → True → `ClientTrainer.update_dataset()` 调用 `poison_data()` → 投毒 DataLoader 存入 `self.local_train_dataset` → **但 `FedMLTrainer.train()` 传入 `self.train_local`（干净数据）→ 投毒失效** ← F-6 所在
- **Scaling**：`FedMLAttacker.is_data_poisoning_attack()` → False → `update_dataset()` 走 else 分支 → `self.local_train_dataset = local_train_dataset`（干净数据，与 `self.train_local` 相同引用）→ F-6 修复后的重定向无副作用
- **Scaling 后门注入**：发生在 `VeriFLTrainer.train()` 内部 `poison_this_round` 分支中，直接修改 batch 张量，与 DataLoader 来源无关

**但注意**：F-6 修复的 `if hasattr(self, 'local_train_dataset') and self.local_train_dataset is not None` 在 `ClientTrainer.__init__` 中 `self.local_train_dataset = None`，**在 `update_dataset()` 调用后此条件对所有客户端（含非恶意）均为 True**。这是安全的（非恶意客户端的 `self.local_train_dataset` 等于传入的干净数据），但条件名不具备区分性——应添加注释说明这一行为。

---

### E-11 🔵 Phase 2 VeriflDefense 注册需修改 `is_defense_on_aggregation()` 白名单

**文档 §4 P2.2 提及**："defend_on_aggregation 的 defense_type 集合中添加 DEFENSE_VERIFL"

**实际白名单**（`fedml_defender.py` L160–161）：

```python
def is_defense_on_aggregation(self):
    return self.is_defense_enabled() and self.defense_type in [
        DEFENSE_SLSGD, DEFENSE_RFA, DEFENSE_WISE_MEDIAN, DEFENSE_GEO_MEDIAN
    ]
```

**如果不将 "verifl" 加入此白名单**，即使 `defense_type=verifl` 且 `is_defense_enabled()=True`，`is_defense_on_aggregation()` 返回 False，`defend_on_aggregation()` 会退化为直接调用 `FedMLAggOperator.agg()`（纯 FedAvg），VeriFL 防御完全不生效。

**建议修正**：在 §4 P2.2 中显式列出需修改的三处代码位置：
1. `constants.py`：新增 `DEFENSE_VERIFL = "verifl"`
2. `fedml_defender.py` `init()`：新增 VeriflDefense 实例化分支
3. `fedml_defender.py` `is_defense_on_aggregation()`：白名单添加 `DEFENSE_VERIFL`

---

### E-12 🟠 `trainer/__init__.py` 未列入 Phase 0 变更表

**遗漏文件**：`python/examples/.../shieldfl/trainer/__init__.py`

**当前内容**：

```python
from .verifl_aggregator import VeriFLAggregator
__all__ = ["VeriFLTrainer", "VeriFLAggregator"]
```

**问题**：§7 文件变更总图中 Phase 0 仅列出 3 个修改文件（`main_fedml_shieldfl.py`、`verifl_trainer.py`、`run_experiment.sh`），遗漏了 `trainer/__init__.py`。该文件包含对 `VeriFLAggregator` 的活跃 import，模块加载时执行，会导致 AC-P0-5 门禁失败。

**决策**：将 re-export 从 `VeriFLAggregator` 改为 `ShieldFLAggregator`，保留 `VeriFLTrainer`（仍在使用）。§7 Phase 0 修改计数从 3 个更新为 4 个。

**修改后**：

```python
from .verifl_trainer import VeriFLTrainer
from .shieldfl_aggregator import ShieldFLAggregator

__all__ = ["VeriFLTrainer", "ShieldFLAggregator"]
```

**为何不整体冻结 `__init__.py`**：`__init__.py` 仅是便捷 re-export，非业务逻辑。旧 re-export 指向即将废弃的 `VeriFLAggregator`，保留会造成 Python 包级别的活跃依赖，与 Phase 0 "切换到 ShieldFLAggregator" 的目标矛盾。

---

### 勘误要点汇总

| 编号 | 严重级 | 分类 | 摘要 | 影响阶段 |
|------|-------|------|------|---------|
| E-1 | 🔴 | 数学推导 | FedAvg 样本加权 ≠ 等权，需条件成立说明 | P1 |
| E-2 | 🔴 | 算法失效 | `server_momentum=0.0` 使 VeriFL Phase 3 为空操作 | P2 |
| E-3 | 🟠 | 参数偏差 | `lambda_reg` 实际 0.01 ≠ 文档 0.1 | P2 |
| E-4 | 🟠 | 审计遗漏 | BaselineAggregator 也覆写 `aggregate()` | P0 |
| E-5 | 🟠 | 接口设计 | ShieldFLAggregator 构造缺参数 | P0 |
| E-6 | 🟡 | 验收条件 | AC-P0-6 标签分布日志不存在 | P0 |
| E-7 | 🟡 | 指标文件 | MetricsCollector aggregator_type 默认值 | P0 |
| E-8 | 🟡 | 审计遗漏 | BaselineAggregator 也 setattr aggregator_type | P0 |
| E-9 | 🔵 | 冻结清单 | 缺少 6 个影响数值的关键参数 | 全局 |
| E-10 | 🔵 | 文档完善 | F-6 修复条件注释不充分 | P0 |
| E-11 | 🔵 | 注册清单 | Phase 2 需修改 3 处代码位置需显式列出 | P2 |
| E-12 | 🟠 | 审计遗漏 | `trainer/__init__.py` re-export `VeriFLAggregator` 未列入 P0 变更表，阻塞 AC-P0-5 | P0 |