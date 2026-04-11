# F-6 调查报告：Label Flipping 投毒数据从未进入训练循环

> **文档性质**：重生推进方案 F-6 缺陷的完整发现路径、论证过程与数值证据  
> **结论**：**确认真实**。M2 LF 实验（commit 02a90c96）中投毒从未生效，24 组实验数据无效  
> **日期**：2026-04-06  
> **前置上下文**：重生推进方案.md §3、LF_实施规格.md、M2_LF_EXPERIMENT_REPORT.md

---

## 目录

- [§1 调查背景与触发点](#1-调查背景与触发点)
- [§2 第一轮：全局架构审计（广度搜索）](#2-第一轮全局架构审计广度搜索)
- [§3 第二轮：代码路径逐级追踪（首次深度验证）](#3-第二轮代码路径逐级追踪首次深度验证)
- [§4 第三轮：LF 实施规格 + commit 对照（二次深度验证）](#4-第三轮lf-实施规格--commit-对照二次深度验证)
- [§5 数值证据分析](#5-数值证据分析)
- [§6 现有测试为何未捕获](#6-现有测试为何未捕获)
- [§7 排除替代假设](#7-排除替代假设)
- [§8 最终结论与影响评估](#8-最终结论与影响评估)

---

## 1. 调查背景与触发点

### 1.1 动机

重生推进方案制定过程中，已知的架构问题集中在 VeriFL 对 FedML Template Method 框架的入侵（F-1 ~ F-5）。在进入 M0 实施阶段前，需要回答一个关键问题：

> "除了 VeriFL 入侵以外，现有架构还有什么一定需要去改的地方？——尤其是那些可能在后续阶段才暴露、并导致海量实验需要重跑的隐藏问题。"

判断标准：只关注**影响学术可用性**的问题，忽略工业级完美性。

### 1.2 审计范围

覆盖 FedML cross_silo 训练管线中与实验正确性直接相关的全部模块：

| 模块 | 审计目标 |
|------|---------|
| 攻击实现（LF、Scaling、Byzantine） | 投毒/后门是否按预期执行 |
| 数据分区（Dirichlet non-IID） | 划分是否正确，RNG 是否隔离 |
| FedAvg 聚合路径 | 聚合算法是否正确 |
| 评估/指标管线 | MA/ASR 是否在干净数据上评估 |
| 安全框架调度器 | 攻击/防御路由是否正确 |
| 种子/可复现性 | 随机状态是否全链路受控 |

---

## 2. 第一轮：全局架构审计（广度搜索）

### 2.1 方法

对上述 6 个模块进行系统性代码阅读，目标是找出所有**影响实验数据有效性**的结构性问题。

### 2.2 发现清单

| 编号 | 发现 | 严重度 | 影响范围 |
|------|------|--------|---------|
| **F-6** | **Label Flipping 投毒数据从未进入训练循环** | **BLOCKING** | **全部 24 组 M2 LF 实验** |
| F-6b | Scaling γ=N 公式假设单攻击者但 K=3 | 条件性 BLOCKING | 已由 D-4 (K=1) 消解 |
| — | 数据分区 Dirichlet 实现 | ✅ 正确 | — |
| — | FedAvg 聚合（`FedMLAggOperator.agg()`） | ✅ 正确 | — |
| — | ASR 评估管线 | ✅ 正确 | — |
| — | 种子控制全链路 | ✅ 正确 | — |

只有 F-6 是新发现的、真正阻塞性的问题。

### 2.3 F-6 初步假设

审计中注意到 FedML 数据投毒管线存在一个对象引用断裂：

- `ClientTrainer.update_dataset()` 调用 `poison_data()` 并将结果存入 `self.local_train_dataset`
- 但 `FedMLTrainer.train()` 传给 `trainer.train()` 的参数是 `self.train_local`——这是在 `FedMLTrainer.update_dataset()` 中从 `train_data_local_dict` 直接取出的原始 DataLoader
- `poison_data()` 返回的是**全新对象**（内部 `torch.cat` 创建新 tensor → `TensorDataset` → 新 `DataLoader`），不修改原始 DataLoader

如果这一假设成立，投毒后的 DataLoader 被存入了一个**从未被读取**的成员变量，训练循环使用的始终是干净数据。

---

## 3. 第二轮：代码路径逐级追踪（首次深度验证）

### 3.1 追踪目标

逐文件、逐行确认完整调用链中"投毒 DataLoader"的去向。

### 3.2 调用链追踪

以下为逐层源码阅读结果。每一步标注文件路径和行号。

#### 步骤 1：`FedMLTrainer.update_dataset(client_index)`

文件：`python/fedml/cross_silo/client/fedml_trainer.py` L50-69

```python
def update_dataset(self, client_index):
    self.client_index = client_index
    # ...
    self.train_local = self.train_data_local_dict[client_index]  # L55: 取原始干净 DataLoader → 对象 (A)
    # ...
    self.trainer.update_dataset(self.train_local, self.test_local, self.local_sample_number)  # L69: 传 (A) 给 ClientTrainer
```

**关键**：`self.train_local` 被设为原始 DataLoader (A)。这个引用在后续 `train()` 中会被直接使用。

#### 步骤 2：`ClientTrainer.update_dataset(local_train_dataset, ...)`

文件：`python/fedml/core/alg_frame/client_trainer.py` L38-48

```python
def update_dataset(self, local_train_dataset, local_test_dataset, local_sample_number):
    if (FedMLAttacker.get_instance().is_data_poisoning_attack()
            and FedMLAttacker.get_instance().is_to_poison_data(client_id=self.id)):
        # W6 fix (D5): only poison train data, keep test data clean
        self.local_train_dataset = FedMLAttacker.get_instance().poison_data(local_train_dataset)  # L42: 创建全新对象 (B)
        self.local_test_dataset = local_test_dataset
    else:
        self.local_train_dataset = local_train_dataset
        self.local_test_dataset = local_test_dataset
```

**关键**：`poison_data(local_train_dataset)` 接收对象 (A)，返回全新对象 (B)。(B) 存入 `self.local_train_dataset`。对象 (A) **未被修改**。

#### 步骤 3：`poison_data()` 确认返回新对象

文件：`python/fedml/core/security/attack/label_flipping_attack.py`（commit 02a90c96 后的版本）

```python
def poison_data(self, local_dataset):
    tmp_local_dataset_x = torch.Tensor([])
    tmp_local_dataset_y = torch.LongTensor([])
    for batch_idx, (data, targets) in enumerate(local_dataset):  # 遍历原始 DataLoader (A)
        tmp_local_dataset_x = torch.cat((tmp_local_dataset_x, data))
        tmp_local_dataset_y = torch.cat((tmp_local_dataset_y, targets.long()))
    
    tmp_y = replace_original_class_with_target_class(...)  # clone + mapping，返回新 tensor
    dataset = TensorDataset(tmp_local_dataset_x, tmp_y)  # 全新 Dataset
    poisoned_data = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)  # 全新 DataLoader (B)
    return poisoned_data
```

**关键**：整个过程通过 `torch.cat`、`clone()`、`TensorDataset`、`DataLoader` 创建全新对象链。原始 DataLoader (A) 的 `Subset` 和底层数据完全不变。

此外，`replace_original_class_with_target_class()` 在 commit 02a90c96 中被修复为 D1 fix：

```python
if isinstance(data_labels, torch.Tensor):
    new_labels = data_labels.clone()    # ← clone，不修改原始
    for orig, tgt in mapping.items():
        new_labels[data_labels == orig] = tgt
    return new_labels
```

#### 步骤 4：`FedMLTrainer.train()` 传入的是什么？

文件：`python/fedml/cross_silo/client/fedml_trainer.py` L71-77

```python
def train(self, round_idx=None):
    self.args.round_idx = round_idx
    tick = time.time()
    self.trainer.on_before_local_training(self.train_local, self.device, self.args)
    self.trainer.train(self.train_local, self.device, self.args)       # L76: 传 self.train_local = 对象 (A)
    self.trainer.on_after_local_training(self.train_local, self.device, self.args)
```

**关键**：传给 `trainer.train()` 的是 `self.train_local`——即步骤 1 中设置的原始干净 DataLoader (A)，**不是** `self.trainer.local_train_dataset`（即对象 (B)）。

#### 步骤 5：`VeriflTrainer.train(train_data, device, args)`

文件：`python/examples/federate/prebuilt_jobs/shieldfl/trainer/verifl_trainer.py` L53

```python
def train(self, train_data, device, args):    # train_data = 参数 = 对象 (A)
    # ... (scaling attack setup) ...
    for epoch in range(num_epochs):
        for batch_idx, (images, labels) in enumerate(train_data):    # ← 遍历 (A)，干净数据
            images, labels = images.to(device), labels.to(device)
            # ... training loop ...
```

**关键**：`train_data` 参数来自 `FedMLTrainer.self.train_local`，即对象 (A)。grep 确认 `verifl_trainer.py` 中对 `self.local_train_dataset` 的引用数量为 **0**。

### 3.3 追踪结论

完整对象生命周期：

```
train_data_local_dict[client_index]  →  对象 (A)  →  FedMLTrainer.self.train_local
                                          │
                                          ↓  传入 poison_data()
                                     遍历 (A) → torch.cat → clone → TensorDataset → DataLoader
                                          │
                                          ↓
                                      对象 (B)  →  ClientTrainer.self.local_train_dataset  ← 死胡同，无人读取
                                      
FedMLTrainer.train()  →  self.trainer.train(self.train_local)  →  train_data = 对象 (A) = 干净数据
```

**投毒后的 DataLoader (B) 存入 `self.local_train_dataset` 后再也没有被使用。训练循环始终使用干净数据。**

### 3.4 攻击路由验证：只有 LF 受影响

进一步确认哪些攻击走 DATA 路径（受 F-6 影响），哪些走 MODEL 路径（不受 F-6 影响）：

| 攻击类型 | 路径 | 调度入口 | F-6 影响 |
|---------|------|---------|---------|
| `label_flipping` | DATA | `ClientTrainer.update_dataset()` → `poison_data()` | **受影响** |
| `model_replacement` | MODEL | `ServerAggregator.on_before_aggregation()` → `attack_model()` | 不受影响 |
| `byzantine` | MODEL | `ServerAggregator.on_before_aggregation()` → `attack_model()` | 不受影响 |

Scaling Attack 在 `verifl_trainer.py` 中通过 inline 后门注入（直接修改 batch 中的 images 和 labels），完全不经过 `poison_data()` 管线，因此不受 F-6 影响。

---

## 4. 第三轮：LF 实施规格 + commit 对照（二次深度验证）

### 4.1 动机

第二轮的追踪基于当前代码状态。需要进一步确认：

1. LF_实施规格.md 的原始设计是否意识到这个问题
2. commit 02a90c96 的实际代码改动是否触及了这条路径
3. 是否有任何"隐藏修复"绕过了这个问题

### 4.2 LF_实施规格.md 的关键声明

#### §5 冻结清单中的声明

规格文档 §5 "不需要改动的部分（冻结清单）" 明确写道：

> | 客户端训练器 | `verifl_trainer.py` | 不 override `update_dataset()`，LF 通过基类 hook 工作 |

这说明规格编写者的**设计假设**是：只要 `ClientTrainer.update_dataset()` 正确执行投毒，`verifl_trainer.py` 无需任何改动——投毒后的数据会自动流入训练循环。

但这个假设是**错误的**。`ClientTrainer.update_dataset()` 确实正确执行了投毒，但投毒结果存入了 `self.local_train_dataset`，而 `FedMLTrainer.train()` 传给 `trainer.train()` 的是来自 `FedMLTrainer` 自己的 `self.train_local`——两个不同层级的不同变量。

#### §10.2 链路全景中的关键注释

规格文档 §10.2 画了完整的执行链路图，在最后关键的一步写道：

```
→ trainer.train(train_data)                 # 用 local_train_dataset 训练
```

注释 `# 用 local_train_dataset 训练` 是**错误的**。`train_data` 的实际来源是 `FedMLTrainer.self.train_local`（干净 DataLoader），不是 `ClientTrainer.self.local_train_dataset`（投毒后的 DataLoader）。规格编写时假设 `FedMLTrainer.train()` 会读取 `self.trainer.local_train_dataset`，但实际代码读的是 `self.train_local`。

#### §6 R2 风险排查的误导

规格文档 §6 R2 排查了 `poison_data()` 是否会累积修改原始数据，结论是"不会"：

> 每轮调用链：`train_data_local_dict[client_index]`（原始 DataLoader）→ `poison_data()` → `torch.cat` 创建新 tensor → `replace_original_class_with_target_class` 在副本上操作 → 返回新 DataLoader。原始 `Subset` 和 `DataLoader` 始终保持干净。

这个分析本身是正确的。但它同时也在说：`poison_data()` 返回的是一个全新对象，原始 DataLoader 不变。如果审阅者在此时追问一步——"既然原始 DataLoader 不变，那训练循环用的是哪个？"——就能发现 F-6。但这个追问没有发生。

### 4.3 commit 02a90c96 的实际改动

通过 `git show 02a90c96c --stat` 确认此 commit 修改了 4 个代码文件：

| 文件 | 改动内容 |
|------|---------|
| `client_trainer.py` | +6/-1：W6 fix (D5)，测试集不再投毒；W3 fix，传入 `client_id=self.id` |
| `label_flipping_attack.py` | +108/-66：完整重写（W1-W5, W7-W8） |
| `utils.py` | +34/-：D1 fix（标签映射），D6 fix（隔离 RNG） |
| `fedml_attacker.py` | +4/-1：适配 `is_to_poison_data` 新签名 |

**关键观察**：commit 没有修改 `FedMLTrainer`（`fedml_trainer.py`）或 `VeriflTrainer`（`verifl_trainer.py`）。投毒数据的"生产"环节（`ClientTrainer.update_dataset()` + `poison_data()`）被正确修复了，但"消费"环节（`FedMLTrainer.train()` → `trainer.train()`）从未被触及，因为规格认为它不需要改动。

### 4.4 检查是否存在隐藏修复

逐一排查是否有其他机制将投毒数据路由进训练循环：

| 可能的路由 | 检查结果 |
|-----------|---------|
| `on_before_local_training()` 是否替换 `train_data`？ | 只处理 FHE 解密，不涉及 DataLoader，不替换 |
| `VeriflTrainer` 是否 override `update_dataset()`？ | 否。grep 确认无 `update_dataset` 定义 |
| `VeriflTrainer.train()` 是否读取 `self.local_train_dataset`？ | 否。grep 确认零引用 |
| `ClientTrainer` 是否有 `__getattr__` 魔法方法？ | 否 |
| `FedMLTrainer.train()` 是否在某处更新 `self.train_local`？ | 否。`self.train_local` 只在 `update_dataset()` 中被设置 |

**没有任何隐藏修复。投毒数据确实被丢弃了。**

---

## 5. 数值证据分析

### 5.1 预测框架

如果 F-6 假设成立（投毒从未生效），则 LF 实验与 clean baseline 之间唯一的差异来源是：

`poison_data()` 在 `ClientTrainer.update_dataset()` 中被调用时，**遍历了原始 DataLoader**（`for batch_idx, (data, targets) in enumerate(local_dataset)`）。这次遍历推进了原始 DataLoader 内部的 `torch.Generator` 状态。虽然训练循环使用的仍是原始 DataLoader (A)，但 (A) 的 shuffle 内部状态已被改变——恶意客户端的第一轮训练将以不同的 batch 顺序看到数据。

但在这个实验中，DataLoader 的 generator 是在 `data_loader.py` 的 `_seeded_dataloader` 中一次性创建的：

```python
generator = torch.Generator()
generator.manual_seed(int(seed))
return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)
```

`poison_data()` 的 `enumerate(local_dataset)` 遍历会消耗这个 generator 的状态。此后训练循环再次 `enumerate(train_data)` 时，shuffle 顺序就不同于 clean baseline。

这种差异的特征是：
- **无系统方向性**：不是"精度一定下降"，而是"精度随机偏移"
- **幅度与训练稳定性成反比**：稳定训练（IID、MNIST）→ 几乎无差异；不稳定训练（极端 non-IID）→ 较大随机偏移
- **可以出现精度上升**：因为这只是 shuffle 噪声，不是真正的对抗扰动

### 5.2 MNIST 结果验证

| α | Absolute Drop (mean) | 预测（shuffle 噪声） | 一致？ |
|---|---------------------|---------------------|--------|
| 0.1 | +0.01% | ≈0（MNIST 对 batch 顺序极不敏感） | ✅ |
| 0.3 | -0.08% | ≈0 | ✅ |
| 0.5 | +0.05% | ≈0 | ✅ |
| 100 | -0.01% | ≈0 | ✅ |

MNIST 全部 4 个 α 配置的 drop 在 ±0.08% 以内，属于纯噪声。且出现了精度**上升**（α=0.1, α=0.5），这与真正的投毒效果（应单调下降）不一致，但与 shuffle 噪声完全一致。

### 5.3 CIFAR-10 IID / 轻度 non-IID 结果验证

| α | Absolute Drop (mean) | 预测 | 一致？ |
|---|---------------------|------|--------|
| 0.5 | -0.30% | ≈0（轻度 non-IID，训练稳定） | ✅ |
| 100 | -0.19% | ≈0（IID，训练最稳定） | ✅ |

这两个配置的 drop 极小（< 0.4%），且逐 seed 数据显示部分 seed 精度上升（α=100 seed=2: +0.19%），完全符合 shuffle 噪声特征。

如果投毒真正生效，即使 LF 是弱攻击，Fang 2025 Table 1-2 在 IID + PMR=20% 下仍报告 ~1-3% drop。PMR=30% 应更强。0.23% 的 drop 不符合投毒生效的预期。

### 5.4 CIFAR-10 极端 non-IID 结果验证（关键）

| α | Absolute Drop (mean) | 逐 seed 明细 | 预测 | 一致？ |
|---|---------------------|-------------|------|--------|
| 0.1 | -4.55% | seed=0: ↓9.43%, **seed=1: ↑4.68%**, seed=2: ↓8.90% | 大方差、双向 | ✅ |
| 0.3 | -2.08% | **seed=0: ↑0.43%**, seed=1: ↓0.62%, seed=2: ↓6.05% | 中等方差、双向 | ✅ |

**α=0.1 seed=1 的精度反升 4.68% 是决定性证据。** 真正的 Label Flipping 投毒——即使效果弱——也不可能使模型精度上升 4.68 个百分点。只有与攻击无关的随机扰动（shuffle 噪声）才能在混沌区（极端 non-IID）产生如此大的双向偏移。

同理，α=0.3 seed=0 精度上升 0.43%，也与真正投毒不一致。

### 5.5 AC-10 FAIL 的重新解释

M2_LF_EXPERIMENT_REPORT.md 的原始解释是：

> "LF 在 PMR=30%、FedAvg 下对 IID/轻度 non-IID 几乎无效——这是正确的学术发现"

这个解释虽然在论文层面是一个合理的现象描述（Fang 2020/2025 确实报告 LF 效果有限），但在我们的实验中，**弱效果的真正原因不是 LF 本身弱，而是投毒数据根本没有进入训练循环**。AC-10 FAIL 不是"学术发现"，而是 F-6 bug 的直接表现。

---

## 6. 现有测试为何未捕获

### 6.1 测试覆盖分析

commit 02a90c96 包含了 `test_lf_correctness.py`（16 项检查，全部 PASS）。逐项分析其覆盖范围：

| 测试 | 验证了什么 | 是否触及 F-6 路径 |
|------|----------|----------------|
| AC-1 (4 checks) | `replace_original_class_with_target_class` 正确性 | ❌ 单元测试，不涉及训练管线 |
| AC-2 (3 checks) | `LabelFlippingAttack` 恶意集合固定性 | ❌ 只测 `__init__` 和 `is_to_poison_data` |
| AC-3 (4 checks) | 每轮投毒客户端正确 | ❌ 只测 `is_to_poison_data` 返回值 |
| AC-4 (1 check) | `poison_data()` 输出 dtype = long | ❌ 只测 `poison_data` 输出 |
| AC-5 (1 check) | `poison_data()` 输出保持 shuffle | ❌ 只测 `poison_data` 输出 |
| AC-6 (2 checks) | `update_dataset` 不投毒测试集 | ❌ 代码审查式检查，不执行训练 |

**核心缺失**：所有测试停在"投毒数据被正确生产"这一环节。没有任何测试检查"投毒数据是否被训练循环实际消费"。

### 6.2 冒烟测试（AC-7 ~ AC-9）的局限

5 轮冒烟测试验证了端到端运行不崩溃、metrics 格式正确、审计日志可追溯。但这些都是"形式正确性"——进程正常退出、文件正常生成——并不检查训练内容。

### 6.3 行为验收（AC-10 ~ AC-12）的误判

AC-10 实际上 FAIL 了（只有 1/4 配置 ≥3%，要求 ≥2），但 LF_实施规格.md §8 AC-10 的"不通过时的处理"流程规定：

> 1. 首先确认 AC-1 ~ AC-9 全部通过
> 2. 与 Fang 2020 / Fang 2025 的已报告趋势对比，确认攻击强度处于合理范围
> 3. 若下降幅度处于合理范围，记录为学术发现

验收报告依据步骤 3 将 AC-10 FAIL 归因为"LF 本身弱"的学术发现，而非代码缺陷。这个归因在表面上合理（论文确实报告 LF 效果有限），但实际上错误——效果有限的真正原因是投毒没有生效。

### 6.4 缺失的关键测试

如果在测试中加入以下任一检查，就能立即发现 F-6：

**方案 A**：在 `VeriflTrainer.train()` 循环内检查 labels 分布

```python
# 在训练循环的第一个 batch 后
label_counts = torch.bincount(labels, minlength=10)
# 如果投毒生效，恶意客户端应看到 [9,8,7,...,0] 映射后的分布
# 如果投毒未生效，分布应与 clean baseline 完全一致
```

**方案 B**：对比训练循环实际使用的 DataLoader 对象 ID

```python
# 在 FedMLTrainer.train() 中
assert id(self.train_local) == id(self.trainer.local_train_dataset), \
    "训练数据与 update_dataset 存储的数据不是同一对象"
```

---

## 7. 排除替代假设

### 7.1 假设："`poison_data()` 修改了原始 DataLoader in-place"

**已排除。** 代码明确显示 `poison_data()` 通过 `torch.cat` + `TensorDataset` + `DataLoader` 创建全新对象链。`replace_original_class_with_target_class()` 使用 `clone()` + 条件赋值，不修改输入 tensor。原始 DataLoader 内部的 `Subset` 引用完全不变。

LF_实施规格.md §6 R2 也独立确认了这一点："原始 `Subset` 和 `DataLoader` 始终保持干净。"

### 7.2 假设："存在某个中间层将 `self.local_train_dataset` 路由回训练循环"

**已排除。** 逐一检查了所有可能的路由：

- `on_before_local_training()`：只处理 FHE 解密
- `VeriflTrainer` 没有 override `update_dataset()`
- `VeriflTrainer.train()` 中对 `self.local_train_dataset` 的引用数为 0（grep 确认）
- `ClientTrainer` 没有 `__getattr__` 等魔法方法
- `FedMLTrainer.train()` 从未更新 `self.train_local`

### 7.3 假设："α=0.1 的 8.36% drop 是真正的投毒效果"

**已排除。** 如果这是真正的投毒效果：
- 所有 3 个 seed 应该都显示精度下降（投毒是确定性的，不应导致精度上升）
- 实际：seed=1 精度**上升** 4.68%，与投毒假设矛盾
- 更合理的解释：α=0.1 下训练对初始条件极度敏感（混沌区），`poison_data()` 遍历 DataLoader 改变了 generator 状态 → shuffle 顺序不同 → 训练轨迹分叉 → 精度随机偏移 ±5-10%

### 7.4 假设："FedML 的其他 cross-silo 模式可能绕过 F-6"

**不适用。** 我们使用的是标准 cross-silo MPI 模式。`FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL` 分支的处理方式类似（`FedMLTrainer.train()` 同样传 `self.train_local`），但不影响当前结论。

---

## 8. 最终结论与影响评估

### 8.1 结论

**F-6 确认真实，不可翻案。**

FedML cross_silo 模式下，数据投毒（data poisoning）管线存在对象引用断裂：
- `ClientTrainer.update_dataset()` 将投毒后的 DataLoader 存入 `self.local_train_dataset`
- `FedMLTrainer.train()` 将**原始**干净 DataLoader（`self.train_local`）传给 `trainer.train()`
- 两者是不同层级的不同成员变量，`poison_data()` 返回新对象后引用彻底分道扬镳

这是 FedML 框架本身的架构缺陷——`ClientTrainer` 和 `FedMLTrainer` 之间存在"变量名相似但实际断裂"的数据流假设。

### 8.2 影响范围

| 实验 | 影响 |
|------|------|
| M2 LF 全部 24 组（commit 02a90c96） | **无效**。观测到的 MTA 变化是 shuffle 噪声，不是投毒效果 |
| M2 Scaling Attack | **不受影响**。Scaling 走 MODEL 路径 + inline 后门注入 |
| 未来任何使用 FedML `is_data_poisoning_attack()` 路径的攻击 | **受影响**，需同步修复 |

### 8.3 修复方案

已纳入重生推进方案 W-M0-8。在 `verifl_trainer.py` 的 `train()` 方法开头加入路由：

```python
def train(self, train_data, device, args):
    # W-M0-8 fix (F-6): route poisoned DataLoader into training loop
    if hasattr(self, 'local_train_dataset') and self.local_train_dataset is not None:
        train_data = self.local_train_dataset
    # ... rest of training ...
```

这将 `ClientTrainer.update_dataset()` 存入的投毒 DataLoader 正确路由进训练循环。

### 8.4 验证方案

已纳入重生推进方案 AC-M0-7。修复后在训练循环内检查恶意客户端的 labels 分布，确认翻转后的标签确实出现在 training loop 中。

### 8.5 LF_实施规格.md 中需要修正的声明

| 位置 | 原文 | 问题 |
|------|------|------|
| §5 冻结清单 | "客户端训练器 \| `verifl_trainer.py` \| 不 override `update_dataset()`，LF 通过基类 hook 工作" | 基类 hook 确实工作了（投毒数据被正确生产），但产物被丢弃而非消费。`verifl_trainer.py` 需要修改以路由投毒数据 |
| §10.2 链路全景 | "`→ trainer.train(train_data) # 用 local_train_dataset 训练`" | `train_data` 实际来源是 `FedMLTrainer.self.train_local`（干净），不是 `ClientTrainer.self.local_train_dataset`（投毒后） |
| §6 R2 | "原始 `Subset` 和 `DataLoader` 始终保持干净。下一轮重新从 dict 取原始数据。" | 分析本身正确，但未意识到"原始保持干净"恰好意味着训练循环用的也是干净数据 |

### 8.6 证据强度总结

| 证据类型 | 内容 | 强度 |
|---------|------|------|
| 代码追踪 | 5 层调用链逐行确认，对象 (A) ≠ 对象 (B)，`verifl_trainer.py` 零引用 `local_train_dataset` | 决定性 |
| Commit 分析 | 02a90c96 未修改 `fedml_trainer.py` 或 `verifl_trainer.py`，投毒"消费"环节从未被触及 | 强 |
| 规格文档 | §5 冻结 `verifl_trainer.py` 的假设 + §10.2 错误注释 = 设计时即存在认知盲区 | 强 |
| MNIST 数值 | 全 8 组 drop ≤ ±0.08%，2 组精度上升 | 强 |
| CIFAR-10 IID/轻 non-IID | α=0.5 drop 0.37%，α=100 drop 0.23%，与投毒预期不符 | 强 |
| CIFAR-10 α=0.1 seed=1 | 精度**上升** 4.68%——真正的投毒不可能导致精度上升 | **决定性** |
| AC-10 FAIL | 只有 1/4 配置 ≥3%（要求 2/4），被误归因为"LF 弱" | 辅助 |
| 隐藏修复排查 | 所有 5 条可能的替代路由全部排除 | 完备 |

---

*本文档为重生推进方案的关键上下文依赖项。后续 M0 实施（W-M0-8）和 M1a 验证（AC-M1a-2/AC-M1a-3）均以此调查结论为前提。*
