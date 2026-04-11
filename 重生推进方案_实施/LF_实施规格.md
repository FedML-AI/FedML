# Label Flipping 攻击 — 学术复现标准文档

> **文档性质**：M2 阶段第一个攻击方法的完整复现标准，可直接交付工程实施  
> **版本**：v2.0 | 2026-04-02  
> **前置依赖**：M1.5 已闭环（24 组 FedAvg 基线已冻结）  
> **威胁模型对齐**：`docs/M2_research/威胁模型.md` §3.2–§3.3  
> **攻击清单对齐**：`docs/M2_research/attack_list.md` 第 ① 项

---

## 目录

- [§1 范围与目标](#1-范围与目标)
- [§2 问题全景：当前代码的 7 个缺陷](#2-问题全景当前代码的-7-个缺陷)
- [§3 设计决策与约束](#3-设计决策与约束)
- [§4 需要完成的工作项（WHAT）](#4-需要完成的工作项what)
- [§5 不需要改动的部分（冻结清单）](#5-不需要改动的部分冻结清单)
- [§6 已排除的风险项](#6-已排除的风险项)
- [§7 实验配置规格](#7-实验配置规格)
- [§8 验收标准](#8-验收标准)
- [§9 实施顺序建议](#9-实施顺序建议)
- [§10 附录](#10-附录)

---

## 1. 范围与目标

### 1.1 本文档覆盖的范围

- 修正 FedML 框架中 Label Flipping (LF) 攻击的全部已知缺陷
- 使修正后的 LF 实现对齐威胁模型 (`威胁模型.md` §3) 的学术语义
- 完成 24 组 LF 实验（2 数据集 × 4 α × 3 seed），产出可用于论文的实验数据

### 1.2 本文档不覆盖的范围

- 客户端采样（`client_num_per_round < client_num_in_total`）——留待后续
- 其他攻击方法（Scaling、Trim、Min-Max 等）——各自有独立规格
- 防御算法的实现——LF 实验使用 FedAvg 无防御作为攻击基线

### 1.3 LF 攻击的学术定义

Label Flipping 是一种**非定向数据投毒攻击**（untargeted data poisoning）。攻击者控制一组固定的恶意客户端 $\mathcal{M}$，在每轮训练前将本地训练集的标签按预设映射全部翻转（如 $0 \to 9, 1 \to 8, ..., 9 \to 0$），使全局模型精度（MA）下降。

本文档的**主参考论文**为：Fang et al., "Local Model Poisoning Attacks to Byzantine-Robust Federated Learning", USENIX Security 2020。  
说明：Tolpegin et al. (2021) 可作为数据投毒背景参考，但本项目中 LF 的无目标攻击复现标准，优先以 Fang 2020 为准。

---

## 2. 问题全景：当前代码的 7 个缺陷

经地毯式代码审阅，当前 LF 实现存在以下 7 个缺陷。按严重程度分为**阻塞项**（不修则实验结果无效）和**必修项**（不修则引入混淆变量或潜在崩溃）。

### 阻塞项（3 个）

#### D1 | 标签映射双重覆盖

| | |
|:--|:--|
| **位置** | `python/fedml/core/security/common/utils.py` → `replace_original_class_with_target_class()` |
| **现象** | 对称映射（0→9 且 9→0）因逐类原地替换导致 class 0 先被改为 9，随后 9 又被改回 0。最终 10 个类中只有 5 个被正确翻转，攻击杀伤力减半。 |
| **根因** | 双层 for 循环直接在 `data_labels` 上读写，后一次迭代覆盖前一次的结果。 |
| **影响** | 攻击效果被严重削弱，实验数据不可用。 |

#### D2 | 恶意客户端选择 ALL-or-NONE

| | |
|:--|:--|
| **位置** | `python/fedml/core/security/attack/label_flipping_attack.py` → `is_to_poison_data()` |
| **现象** | 在 cross-silo MPI 模式下，每轮要么所有 10 个客户端全部投毒，要么全部不投毒，永远不会出现"3 个投毒 + 7 个正常"。 |
| **根因** | 每个 MPI 进程独立创建 `LabelFlippingAttack` 实例，所有实例的 `counter` 从 0 开始同步递增 → 相同的 `np.random.seed(counter)` → 相同的 `np.random.random()` → 相同的决策。 |
| **影响** | 违背威胁模型 §3.2.1（固定子集 $\mathcal{M}$），PMR 语义完全错误。 |

#### D3 | poison_data() 标签 dtype 被降级为 float32

| | |
|:--|:--|
| **位置** | `python/fedml/core/security/attack/label_flipping_attack.py` → `poison_data()` |
| **现象** | 函数开头 `tmp_local_dataset_y = torch.Tensor([])` 创建 float32 空 tensor。随后 `torch.cat((tmp_local_dataset_y, targets))` 将 DataLoader 输出的 long（int64）labels 与 float32 拼接。根据 PyTorch 类型提升规则，结果可能被转为 float 或直接报错（取决于 PyTorch 版本）。 |
| **根因** | 用 `torch.Tensor([])` 作为累加器起点，dtype 为 float32，而非 `torch.LongTensor([])`。 |
| **影响（若 labels 变成 float32）** | `nn.CrossEntropyLoss` 对 float 类型的 target 会按 **soft label** 语义处理（概率分布），而非 class index 语义。训练行为完全不同于预期。在某些 PyTorch 版本下可能直接崩溃。 |

### 必修项（4 个）

#### D4 | poison_data() 丢失 DataLoader shuffle

| | |
|:--|:--|
| **位置** | `label_flipping_attack.py` → `poison_data()` 末尾 |
| **现象** | 原始 DataLoader 由 `_seeded_dataloader()` 创建，`shuffle=True`。`poison_data()` 重建 DataLoader 时使用 `DataLoader(dataset, batch_size=self.batch_size)`，没有传 `shuffle=True`。 |
| **影响** | 恶意客户端训练时数据不 shuffle，梯度估计有偏。良性客户端有 shuffle。这引入了一个与攻击无关的**混淆变量**——MA 下降中，有多少来自 label flipping，有多少来自丢失 shuffle，无法分离。 |

#### D5 | 测试集被投毒

| | |
|:--|:--|
| **位置** | `python/fedml/core/alg_frame/client_trainer.py` → `update_dataset()` |
| **现象** | 当 `is_to_poison_data()` 返回 True 时，`local_train_dataset` 和 `local_test_dataset` 都被传入 `poison_data()` 进行标签翻转。 |
| **影响** | 对于 LF 攻击，客户端本地测试集标签不应被翻转。当前不影响全局 MA 评估（server 端用独立 test_loader），也不影响实际实验（因为 `test_on_clients` 未配置，客户端侧 test 不执行）。但如果未来启用客户端侧评估，会导致指标失真。 |

#### D6 | 恶意客户端生成使用全局 numpy 随机状态

| | |
|:--|:--|
| **位置** | `python/fedml/core/security/common/utils.py` → `get_malicious_client_id_list()` |
| **现象** | 函数用 `np.random.seed(random_seed)` 设置**全局** numpy 随机状态。任何此后使用 `np.random`（而非 `np.random.default_rng`）的代码都会受到影响。 |
| **影响** | 我们的 `data_loader.py` 中 `_split_noniid` 使用 `np.random.default_rng(seed)`（隔离 RNG），所以数据划分不受影响。但如果其他代码（如未来新增的模块）使用全局 `np.random`，可能产生难以追踪的不可复现行为。 |

#### D7 | 轮次计数在 cross-silo 模式下失效

| | |
|:--|:--|
| **位置** | `label_flipping_attack.py` → `get_ite_num()` |
| **现象** | `floor(counter / client_num_per_round)` 假设 counter 跨所有客户端递增（即每轮 counter 增加 N 次）。但在 MPI 模式下，每个进程有独立的 counter，每轮只增加 1 次。导致计算出的轮次号 = `floor(actual_round / N)`，严重滞后。 |
| **影响** | `poison_start_round_id` 和 `poison_end_round_id` 失效。例如配置 `poison_start_round_id=5`，实际要到第 50 轮才开始投毒（N=10 时）。由于我们使用全程投毒（start=0, end=comm_round-1），起始轮未受影响，但结束轮条件可能提前截断。工程修复 D2 时如果改用固定集合判断，轮次追踪机制也需一并修正或简化。 |

---

## 3. 设计决策与约束

### 3.1 决策一：恶意客户端选择机制

**结论**：复用 FedML 已有的 `get_malicious_client_id_list()` 模式（seed → 固定集合），而非发明新机制。

**依据**：
- FedML 原作者在 `utils.py` L104 的注释：*"make sure for each comparison, we are selecting the same clients each round"*
- `EdgeCaseBackdoorAttack` 已使用此模式，成熟可靠
- 与威胁模型 §3.2.1 的固定 $\mathcal{M}$ 完美对齐
- 同一 `random_seed` 保证跨实验可复现；不同 seed 保证统计独立性

### 3.2 决策二：最大程度复用 FedML 配置架构

**结论**：不新增、不重命名任何 YAML 配置字段。

**依据**：
- `run_experiment.sh` 已为 LF 预留完整字段：`original_class_list`、`target_class_list`、`ratio_of_poisoned_client`
- FedML 的 YAML → `args.xxx` 自动映射完全够用
- 攻击路由 `fedml_attacker.py` → `LabelFlippingAttack(args)` 已注册

### 3.3 决策三：`random_seed` 的多用途复用

**结论**：同一个 `random_seed`（来自 `--seed` 参数）同时控制：
1. 数据的 non-IID 划分（`_split_noniid` 中的 `np.random.default_rng(seed)`）
2. 恶意客户端集合生成（`get_malicious_client_id_list` 中的种子）
3. DataLoader 的 shuffle 顺序（`_seeded_dataloader` 中的 `torch.Generator().manual_seed(seed + offset)`）

**风险分析**：三者虽共享同一 seed **数值**，但使用**不同的 RNG 实例**（numpy default_rng / numpy global / torch Generator），因此互不干扰。但 D6 中指出的全局 numpy 状态污染需要修复——改用 `np.random.default_rng` 隔离。

**额外约束**：恶意集合生成必须用**专用的、与数据划分不同的 RNG 实例**，即使输入种子值相同。这保证了：
- 改变数据划分方式不会改变恶意集合
- 改变恶意集合算法不会影响数据划分

### 3.4 决策四：client_id 传递方式

**现状**：`ClientTrainer.update_dataset()` 调用 `FedMLAttacker.is_to_poison_data()` 时不传 `client_id`。但 trainer 自身有稳定的 `self.id`（由 `fedml_trainer_dist_adapter.py` L35 的 `model_trainer.set_id(client_index)` 设置，0-indexed）。

**结论**：需要让 LF 攻击类在 `is_to_poison_data()` 调用时获知当前 client_id。具体传递方式（参数传递 / setter 方法 / 访问 trainer 引用）由工程同学决定。关键约束是：**client_id 必须是稳定的 0-indexed 值，与 `set_id(client_index)` 设置的值一致。**

### 3.5 与威胁模型的逐条对齐

| 威胁模型条款 | LF 实现要求 | 修复后状态 |
|:------------|:-----------|:----------|
| §3.2.1: 攻击者控制固定集合 $\mathcal{M} \subseteq \{C_1,...,C_N\}$, $\|\mathcal{M}\| = K$ | 恶意集合跨轮不变 | ✅ 修 D2 后由 seed 一次性确定 |
| §3.2.1: 在数据层面操纵本地训练集 | 标签正确翻转 | ✅ 修 D1 后 10 类全部正确翻转 |
| §3.1: 服务器采样 $\mathcal{C}^{(t)}$ | 兼容 $n < N$ | ✅ 固定集合模式天然兼容 |
| §3.3.1: 非定向投毒 → 降 MA | 全标签反转 | ✅ 攻击语义正确 |
| §3.2.2: 攻击者无法访问 $D_{val}$ | LF 不接触 server 数据 | ✅ 纯 client 端操作 |

---

## 4. 需要完成的工作项（WHAT）

以下列出所有需要工程实施的工作项。每项只说明**做什么**和**为什么**，不规定**怎么做**。

### W1 | 修正标签映射函数

| | |
|:--|:--|
| **修什么** | `utils.py` → `replace_original_class_with_target_class()` |
| **目标** | 输入任意映射 `original→target`，所有标签被正确翻转一次且仅一次 |
| **核心约束** | 不能在同一份 labels 上边读边写（避免覆盖）；函数签名不变（不影响 `edge_case_backdoor_attack.py` 等其他调用方） |
| **解决 D1** | ✅ |

### W2 | 重写 LF 恶意客户端选择为固定集合模式

| | |
|:--|:--|
| **修什么** | `label_flipping_attack.py` → `__init__()` 和 `is_to_poison_data()` |
| **目标** | 初始化时一次性生成固定恶意 ID 集合 $\mathcal{M}$，运行时判断 `current_client_id ∈ $\mathcal{M}$` |
| **核心约束** | 复用现有字段 `ratio_of_poisoned_client` 和 `random_seed`，不新增 YAML 字段；使用隔离的 RNG 实例（不污染全局 numpy 状态） |
| **解决 D2, D6** | ✅ |

### W3 | 建立 client_id 传递通道

| | |
|:--|:--|
| **修什么** | `client_trainer.py` → `update_dataset()` 和/或 `fedml_attacker.py` |
| **目标** | 在 `is_to_poison_data()` 被调用时，攻击类能获取到当前客户端的 0-indexed `client_id` |
| **核心约束** | `client_id` 的值必须等于 `trainer.id`（来自 `set_id(client_index)`）；不能依赖 MPI rank 等进程级信息（为兼容未来 cross-device 模式） |
| **解决 D2**（配合 W2） | ✅ |

### W4 | 修正 poison_data() 的 label dtype 保持

| | |
|:--|:--|
| **修什么** | `label_flipping_attack.py` → `poison_data()` |
| **目标** | 确保 poison 后的 labels tensor dtype 与原始 DataLoader 输出的 dtype 一致（通常为 `torch.long` / `int64`） |
| **核心约束** | 不能用 `torch.Tensor([])` 作为 float32 累加器起点；最终传入 `TensorDataset` 的 labels 必须是 `long` 类型 |
| **解决 D3** | ✅ |

### W5 | 修正 poison_data() 重建 DataLoader 保持 shuffle

| | |
|:--|:--|
| **修什么** | `label_flipping_attack.py` → `poison_data()` |
| **目标** | 重建的 DataLoader 与原始 DataLoader 保持相同的 `shuffle` 行为 |
| **核心约束** | 消除恶意客户端与良性客户端在数据 shuffle 上的差异，排除混淆变量 |
| **解决 D4** | ✅ |

### W6 | 去除测试集投毒

| | |
|:--|:--|
| **修什么** | `client_trainer.py` → `update_dataset()` |
| **目标** | 当 LF 攻击启用时，只对 `local_train_dataset` 投毒，`local_test_dataset` 保持干净 |
| **核心约束** | 改动量最小——只需去掉对 test_dataset 调用 `poison_data()` 的那一行 |
| **解决 D5** | ✅ |

### W7 | 修正轮次追踪机制

| | |
|:--|:--|
| **修什么** | `label_flipping_attack.py` → 轮次计数逻辑 |
| **目标** | 在 cross-silo MPI 模式下，`poison_start_round_id` 和 `poison_end_round_id` 按真实轮次生效 |
| **核心约束** | 如果改为固定集合模式（W2）后不再需要按轮次概率判断，则轮次追踪可大幅简化——只需为 round window 功能保留正确的当前轮次号。轮次号的来源可考虑从外部传入（如 `args.round_idx`），或由 counter 直接等于轮次号（cross-silo 下每进程每轮 counter +1，无需除以 N）。 |
| **解决 D7** | ✅ |

### W8 | 补充投毒审计日志

| | |
|:--|:--|
| **修什么** | `label_flipping_attack.py` 和/或 `client_trainer.py` |
| **目标** | 日志中可追溯：(a) 恶意客户端 ID 列表（初始化时打印一次），(b) 每轮哪些客户端执行了投毒，(c) 恶意客户端总数 |
| **核心约束** | 使用 `logging.info` 级别；日志格式不限，但必须包含足够信息回答上述三个问题 |

---

## 5. 不需要改动的部分（冻结清单）

以下组件经代码审阅确认完全正确，**不要改动**：

| 组件 | 文件 | 说明 |
|:-----|:-----|:-----|
| 攻击类型常量 | `constants.py` | `ATTACK_LABEL_FLIPPING = "label_flipping"` 已注册 |
| 攻击路由 | `fedml_attacker.py` → `init()` | LF 分支路由正确 |
| 攻击分类 | `fedml_attacker.py` → `is_data_poisoning_attack()` | LF 正确归类为 data poisoning |
| 脚本参数解析 | `run_experiment.sh` → `--attack` 分支 | YAML 生成逻辑正确 |
| YAML 字段名 | `run_experiment.sh` → YAML 模板 | 所有 LF 字段名与构造函数匹配 |
| 服务器聚合 | `verifl_aggregator.py` → `on_before_aggregation()` | LF 不走 model_attack 分支 |
| 客户端训练器 | `verifl_trainer.py` | 不 override `update_dataset()`，LF 通过基类 hook 工作 |
| 服务器评估 | `baseline_aggregator.py` / `verifl_aggregator.py` → `test()` | 使用独立的全局 test_loader，不受客户端投毒影响 |
| 指标采集 | `eval/metrics.py` → `MetricsCollector` | JSONL 格式正确，字段完整，已包含 `attack_type`、`asr` 等 |
| 数据加载 | `data/data_loader.py` → `load_shieldfl_data()` | non-IID 划分使用隔离 RNG (`default_rng`)，不受攻击模块影响 |
| 客户端 ID 设置 | `fedml_trainer_dist_adapter.py` → `set_id(client_index)` | 0-indexed，稳定，可作为投毒判断依据 |

---

## 6. 已排除的风险项

以下是审阅中主动排查但确认**不构成问题**的点，记录在此防止重复排查。

### R1 | random_seed 是否导致数据划分与恶意集合相关？

**结论：不会。**

虽然 `random_seed` 数值相同，但数据划分用 `np.random.default_rng(seed)`（隔离 RNG），恶意集合生成用另一个 RNG 实例。两者的随机序列完全独立。

### R2 | poison_data() 是否会累积修改原始数据？

**结论：不会。**

每轮调用链：`train_data_local_dict[client_index]` （原始 DataLoader）→ `poison_data()` → `torch.cat` 创建新 tensor → `replace_original_class_with_target_class` 在副本上操作 → 返回新 DataLoader。原始 `Subset` 和 `DataLoader` 始终保持干净。下一轮重新从 dict 取原始数据。

### R3 | FedMLAttacker 单例是否跨进程共享？

**结论：不共享。**

MPI 模式下每个进程有独立的 Python 解释器，`_attacker_instance = None` 是 class-level 变量，每个进程各自实例化。进程间完全隔离。

### R4 | attack_prob 是否引入额外随机性？

**结论：当前不会。**

`FedMLAttacker.is_attack_enabled()` 中的 `random.random() <= self.attack_prob` 检查在 `attack_prob=1`（默认值，我们不设置此字段）时恒为 True。无额外随机性。但若未来有人设置 `attack_prob < 1`，会引入一层不可控的非确定性。YAML 模板中不添加此字段即可。

### R5 | 客户端侧测试（test_on_clients）是否受影响？

**结论：当前不受影响。**

`run_experiment.sh` 生成的 YAML 未设置 `test_on_clients` 字段 → `fedml_client_master_manager.py` 的 `__test()` 方法 `hasattr` 检查失败 → 直接 return → 客户端侧测试不执行。但 W6（去除测试集投毒）仍应执行，以防未来启用此功能。

### R6 | server 端全局评估是否使用了干净数据？

**结论：是的。**

`baseline_aggregator.py` 和 `verifl_aggregator.py` 的 `test()` 方法使用 `test_data`（来自 server 端的全局 `test_loader`，在 `load_shieldfl_data()` 中创建）。这条数据链路完全独立于客户端投毒链路。MA 和 test_loss 的评估基于干净数据，正确。

### R7 | non-IID 分布是否影响 LF 的正确性？

**结论：不影响正确性，但影响攻击效果。**

在极端 non-IID（α=0.1）下，某些客户端可能只持有少数几个类别的数据。LF 全标签反转后，这些客户端的训练集会包含"原本不存在的类别标签"。这是 LF 在 non-IID 场景下的预期行为，不是 bug。

α 对 LF 效果的影响是学术研究的一部分——这正是为什么实验矩阵包含 4 个 α 值。

### R8 | `batch_size` 在 poison_data() 中是否一致？

**结论：一致。**

`LabelFlippingAttack.__init__` 读取 `args.batch_size`，`_seeded_dataloader` 也读取同一个 `args.batch_size`。两者来自同一 YAML 配置，值必然一致。

---

## 7. 实验配置规格

### 7.1 攻击配置字段（YAML train_args 段）

所有字段均复用 FedML 现有命名，不新增。

```yaml
enable_attack: true
attack_type: "label_flipping"

# 标签映射 (10 类全反转，按 Fang 2020 在 CIFAR-10 / MNIST 设定对齐)
original_class_list: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
target_class_list: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

# 恶意客户端比例 (attack_list.md §5 默认 30%)
ratio_of_poisoned_client: 0.3

# 投毒轮次窗口 (全程投毒，使用代码默认值)
# poison_start_round_id: 0        # 默认 0
# poison_end_round_id: <rounds-1> # 默认 comm_round - 1

# 以下字段在模板中存在但 LF 不读取，保留不删
eval_asr: false                    # LF 是 untargeted，不评 ASR
byzantine_client_num: 3            # LF 不使用
attack_mode: "flip"                # LF 不使用
target_label: 0                    # LF 不使用
trigger_size: 3                    # LF 不使用
trigger_value: 1.0                 # LF 不使用
```

### 7.2 参数冻结说明

以下参数可一次冻结，不存在返工风险：

| 参数 | 冻结值 | 不返工的理由 |
|:-----|:------|:-----------|
| 标签映射 | `[0..9] → [9..0]` | 对齐 Fang 2020 的无目标 LF 复现实验设定 |
| PMR | 0.3 | 对齐 `attack_list.md` §5 |
| 投毒窗口 | 全程 | LF 标准设定，无"只攻前半段"的变体 |
| 恶意集合由 seed 决定 | ✓ | 同 seed 可复现，换 seed 自动变更 |
| eval_asr | false | LF 是 untargeted 攻击，ASR 不适用 |

唯一可能返工的场景：未来决定 PMR 不是 0.3。但那只需改 `--pmr` 脚本参数，不影响代码。

### 7.3 实验矩阵

| 变量 | 值域 | 说明 |
|:-----|:-----|:-----|
| `--seed` | {0, 1, 2} | 控制数据划分 + 恶意集合 |
| `--alpha` | {0.1, 0.3, 0.5, 100} | Dirichlet 异构度 |
| `--dataset` | {cifar10, mnist} | |
| `--model` | ResNet18 (cifar10), LeNet5 (mnist) | M1 冻结 |
| `--rounds` | 100 (cifar10), 50 (mnist) | M1 冻结 |
| `--epochs` | 1 | M1 冻结 |
| `--batch_size` | 64 | M1 冻结 |
| `--lr` | 0.01 | M1 冻结 |
| `--pmr` | 0.3 | |

$$\text{总实验数} = 2 \text{ datasets} \times 4\ \alpha \times 3 \text{ seeds} = 24$$

每组实验对应的 M1.5 基线已存在（同 dataset / α / seed 的 FedAvg 无攻击结果）。

---

## 8. 验收标准

分三个层次，每一层的通过是下一层的前置条件。

### 第一层：代码正确性（上线阻塞项）

AC-1 ~ AC-6 全部通过后，方可开始正式实验。

---

**AC-1 | 标签映射正确性**

调用修正后的 `replace_original_class_with_target_class`：
- 输入：`labels = [0,1,2,3,4,5,6,7,8,9]`（long 类型），映射 `[0..9] → [9..0]`
- 期望输出：`[9,8,7,6,5,4,3,2,1,0]`

**通过条件**：10 个类全部正确翻转，无双重覆盖。输出 dtype 与输入一致。

---

**AC-2 | 恶意客户端集合固定性**

配置 `random_seed=0, client_num_in_total=10, ratio_of_poisoned_client=0.3`。

**通过条件**：
- 两次独立初始化输出完全一致
- 集合大小 = `ceil(10 × 0.3) = 3`
- 更换 `random_seed=1` 后，集合与 seed=0 不同

---

**AC-3 | per-round 投毒正确性**

10 客户端全参与，运行 5 轮。

**通过条件**：
- 每轮恰好 3 个恶意客户端投毒，7 个不投毒
- 5 轮的恶意集合完全一致
- 不出现 ALL-or-NONE

---

**AC-4 | 标签 dtype 保持**

投毒后从 DataLoader 迭代取出 labels。

**通过条件**：labels dtype 为 `torch.long`（int64），与未投毒客户端的 labels dtype 一致。

---

**AC-5 | DataLoader shuffle 保持**

在两次迭代投毒后 DataLoader 时，观察 batch 顺序。

**通过条件**：两次迭代的 batch 顺序不同（证明 shuffle 生效）。或者通过代码审查确认重建的 DataLoader 包含 `shuffle=True` 参数。

---

**AC-6 | 测试集干净**

检查恶意客户端的 `local_test_dataset` 标签分布。

**通过条件**：与良性客户端一致，未被翻转。（注：当前 `test_on_clients` 未启用，此项通过代码审查确认 `update_dataset()` 不对 test 调用 `poison_data()` 即可。）

---

### 第二层：链路正确性（集成测试）

---

**AC-7 | 端到端运行不崩溃**

命令：
```bash
bash scripts/run_experiment.sh \
  --model ResNet18 --dataset cifar10 \
  --attack label_flipping --defense none --aggregator fedavg \
  --pmr 0.3 --alpha 0.5 --seed 0 \
  --rounds 5 --clients 10 --epochs 1 --batch_size 64 --gpu
```

**通过条件**：
- 进程正常退出（exit code 0）
- 生成 JSONL metrics 文件
- 文件中 round 0 ~ round 4 均有 `test_accuracy` 和 `test_loss`

---

**AC-8 | metrics 文件内容正确**

**通过条件**：
- `attack_type` 字段值为 `"label_flipping"`
- `asr` 字段为 `null`（LF 不评 ASR）
- `random_seed`、`pmr`、`alpha` 等元信息正确记录

---

**AC-9 | 投毒审计日志可追溯**

从日志中能回答：
1. 恶意客户端集合是哪些 ID？
2. 每轮哪些客户端执行了投毒？
3. 恶意客户端总数？

**通过条件**：三个问题均可从日志明确回答。

---

### 第三层：实验行为验收（学术可用性）

在完整实验配置下运行（100 轮 CIFAR-10 / 50 轮 MNIST），对比 M1.5 基线。

---

**AC-10 | 攻击产生非零损害**

$$\text{Relative MA Drop} = \frac{\text{MA}_{baseline} - \text{MA}_{LF}}{\text{MA}_{baseline}} \times 100\%$$

其中 MA 取最终轮 3-seed 均值。

**通过条件**：CIFAR-10 的至少 2 个 α 配置上，Relative MA Drop ≥ 3%。

**不通过时的处理**：
1. 首先确认 AC-1 ~ AC-9 全部通过
2. 与 Fang 2020 / Fang 2025 的已报告趋势对比，确认攻击强度处于合理范围
3. 若下降幅度处于合理范围，记录为学术发现
4. 若完全无下降，排查 `poison_data` 是否真正传入了翻转后的 DataLoader

---

**AC-11 | 方向一致性**

**通过条件**：在满足 AC-10 的 α 配置上，3 个 seed 中至少 2 个 seed 的 MA 低于对应基线。

---

**AC-12 | MNIST 预期行为**

根据论文和历史数据，MNIST 对 LF 近乎免疫。

**通过条件**：MNIST 实验正常完成并记录结果。Relative MA Drop < 1% 视为"攻击无效但实现正确"的学术发现，不阻塞验收。

---

## 9. 实施顺序建议

```
Phase A — 代码修改（W1 ~ W8）
  Step 1: 修正标签映射函数 (W1)                   → 验 AC-1
  Step 2: 重写恶意客户端选择 + 传递 client_id      → 验 AC-2, AC-3
          (W2, W3, W7 可一起做)
  Step 3: 修正 poison_data() dtype + shuffle       → 验 AC-4, AC-5
          (W4, W5 可一起做)
  Step 4: 去除测试集投毒 (W6)                      → 验 AC-6
  Step 5: 补充审计日志 (W8)                        → 验 AC-9

Phase B — 冒烟测试
  Step 6: 5 轮短实验端到端跑通                     → 验 AC-7, AC-8

Phase C — 正式实验
  Step 7: CIFAR-10 × 4α × 3seed = 12 组           → 验 AC-10, AC-11
  Step 8: MNIST × 4α × 3seed = 12 组              → 验 AC-12
```

---

## 10. 附录

### 10.1 恶意客户端集合与客户端采样的兼容性

当前全参与模式：

$$\mathcal{C}^{(t)} = \{C_1, ..., C_N\},\quad \mathcal{A}^{(t)} = \mathcal{M}$$

未来客户端采样模式：

$$\mathcal{C}^{(t)} \subset \{C_1, ..., C_N\},\quad |\mathcal{C}^{(t)}| = n < N,\quad \mathcal{A}^{(t)} = \mathcal{M} \cap \mathcal{C}^{(t)}$$

本文档的设计保证：切换到采样模式时，只需修改 FL 框架的客户端选择逻辑，LF 攻击类本身零修改。

### 10.2 代码审阅链路全景

完整执行链从配置到训练：

```
run_experiment.sh
  → 生成 YAML (train_args 含攻击字段)
    → mpirun -np 11 (1 server + 10 clients)
      → 每个 client 进程:
        TrainerDistAdapter.__init__
          → model_trainer.set_id(client_index)    # trainer.id = 0..9
          → ClientTrainer.__init__
            → FedMLAttacker.get_instance().init(args)
              → LabelFlippingAttack(args)         # 生成恶意集合 M
      → 每轮:
        handle_message_receive_model_from_server
          → trainer_dist_adapter.update_dataset(client_index)
            → FedMLTrainer.update_dataset(client_index)
              → self.train_local = train_data_local_dict[client_index]  # 干净 DataLoader
              → trainer.update_dataset(train_local, test_local, ...)
                → if is_data_poisoning_attack() and is_to_poison_data():
                    → poison_data(train_local)        # 翻转标签 → 新 DataLoader
                    → self.local_train_dataset = ...  # 投毒后的
                    → self.local_test_dataset = ...   # 保持干净 (W6)
          → trainer.train(train_data)                 # 用 local_train_dataset 训练
          → send_model_to_server()
```

### 10.3 缺陷-工作项-验收标准 追溯矩阵

| 缺陷 | 工作项 | 验收标准 |
|:-----|:------|:---------|
| D1 标签双重覆盖 | W1 | AC-1 |
| D2 ALL-or-NONE 选择 | W2, W3 | AC-2, AC-3 |
| D3 label dtype 降级 | W4 | AC-4 |
| D4 丢失 shuffle | W5 | AC-5 |
| D5 测试集投毒 | W6 | AC-6 |
| D6 全局 numpy 状态污染 | W2（使用隔离 RNG） | AC-2 |
| D7 轮次计数失效 | W7 | AC-3 |
| — 审计日志 | W8 | AC-9 |

### 10.4 推进本工作的参考文献

以下文献建议与本文档一起交付工程同学，作为实现边界、实验预期和结果解释的共同参考。

1. Fang et al., "Local Model Poisoning Attacks to Byzantine-Robust Federated Learning", USENIX Security 2020。
用途：本项目 LF 无目标攻击的主参考文献；用于校准攻击定义、实验预期和结果解释。

2. Fang 等 - 2025 - Do We Really Need to Design New Byzantine-robust Aggregation Rules.md
用途：用于对照近年实验结论，判断 LF 在不同聚合与设置下的相对强弱是否合理。

3. Tolpegin et al., "Data Poisoning Attacks Against Federated Learning Systems", 2021。
用途：作为 LF / 数据投毒的背景补充参考，不作为本项目无目标 LF 的主标准。

4. 威胁模型.md
用途：工程实现时的语义边界定义，尤其是固定恶意集合 $\mathcal{M}$、服务器干净验证集、untargeted attack 目标等。

5. attack_list.md
用途：本阶段攻击清单与默认参数来源，尤其是 PMR 和攻击优先级。

6. M1.5_EXPERIMENT_REPORT.md
用途：LF 实验的 FedAvg 无攻击基线来源，用于 MA drop 对比。
