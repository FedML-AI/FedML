# Scaling Attack 复现实施定稿

> 文档性质：交付工程实施的唯一有效定稿
> 适用范围：FedML + ShieldFL 当前代码库中的 Scaling Attack / Model Replacement Backdoor Attack 复现
> 日期：2026-04-04

---

## 1. 目标与冻结边界

本文档只定义最终落地方案，不保留版本演化信息。工程实现、实验执行、验收判定均以本文档为准。

本项目中 Scaling Attack 的目标不是复刻 Bagdasaryan 原文的完整 100-client 实验壳，而是在 **当前仓库已经冻结的 M1.5 实验载体** 上，严谨复现其不可删减的攻击核心语义：

1. 恶意客户端必须进行后门训练
2. 服务端必须对恶意客户端更新执行缩放
3. 训练时 trigger 注入逻辑必须与 ASR 评估逻辑完全一致
4. 攻击实验必须与 M1.5 / LF 的主实验骨架直接可比

因此，本项目的 Scaling 复现载体冻结为：

| 维度 | 冻结值 |
|:-----|:------|
| 训练模式 | cross-silo |
| 通信后端 | MPI |
| 客户端总数 | 10 |
| 每轮参与客户端数 | 10 |
| 聚合器 | FedAvg |
| CIFAR-10 模型 | ResNet18 |
| MNIST 模型 | LeNet5 |
| CIFAR-10 通信轮数 | 100 |
| MNIST 通信轮数 | 50 |
| local epochs | 1 |
| batch_size | 64 |
| learning_rate | 0.01 |
| server_lr | 1.0 |
| weight_decay | CIFAR-10=1e-4, MNIST=0 |
| momentum | 0.9 |
| alpha 网格 | {0.1, 0.3, 0.5, 100} |
| seed 网格 | {0, 1, 2} |
| clean baseline | M1.5 冻结结果 |

上表不可改。若未来需要做 100-client 社区对标，那是新实验线，不属于本文档。

---

## 2. 攻击最终定义

本项目中的 Scaling Attack 定义如下：

1. 恶意客户端集合固定为前 K 个客户端，即 client_id ∈ {0, 1, ..., K-1}
2. 恶意客户端**仅在攻击轮次（`attack_training_rounds`）**进行后门训练；非攻击轮次恶意客户端行为与良性客户端完全一致（不注入 trigger，不做任何特殊处理）
3. 在攻击轮次的本地训练中，恶意客户端把每个 batch 中固定 20 张图片改造成后门样本
4. 后门样本的 trigger 打在右下角 3×3 区域，像素值填充为 1.0（在归一化空间中）
5. 后门样本标签统一改为 target_label=0
6. 恶意客户端完成本地训练后，服务端在同一攻击轮次对其模型参数执行缩放：

$$W_i' = \gamma (W_i - G) + G$$

其中：

- $W_i$ 为恶意客户端本轮上传模型
- $G$ 为聚合前的全局模型
- $\gamma$ 为缩放系数

本项目的默认缩放系数冻结为：

$$\gamma = \text{client\_num\_in\_total} = 10$$

由于本文档冻结为全参与模式，因此有：

$$\text{client\_num\_per\_round} = \text{client\_num\_in\_total} = 10$$

实现中必须在 server 攻击类 `ModelReplacementBackdoorAttack.__init__` 中显式断言 `client_num_per_round == client_num_in_total`；若未来改为部分参与，本文档失效。

---

## 3. 最终参数方案

### 3.1 攻击参数

| 参数 | 最终值 | 说明 |
|:-----|:------|:-----|
| attack_type | model_replacement | 保持 FedML 现有常量 |
| eval_asr | true | Scaling 必须评估 ASR |
| pmr | 0.3 | 对齐当前项目攻击主线 |
| byzantine_client_num | 3 | 10 客户端 × 0.3，向上取整 |
| malicious_client_ids | [0, 1, 2] | 固定，禁止随机 |
| scale_gamma | 10 | 默认等于 client_num_in_total |
| target_label | 0 | 统一冻结 |
| trigger_size | 3 | 右下角 3×3 |
| trigger_value | 1.0 | 与现有 ASR 评估一致 |
| backdoor_per_batch | 20 | 固定，不再使用 poison_ratio |
| attacker_epochs | null | null 表示跟随全局 epochs=1 |
| attacker_lr | null | null 表示跟随全局 learning_rate=0.01 |
| attacker_weight_decay | null | null 表示跟随全局 weight_decay |
| attacker_noise_sigma | 0 | 不加高斯噪声 |

### 3.2 攻击轮次窗口

主实验不采用“从第 0 轮开始每轮攻击”，也不采用“只打一轮”的极端设置，而采用 **末段 5 轮攻击窗口**，原因是：

1. 过早攻击会把 targeted backdoor 变成粗暴破坏 clean task 的训练扰动
2. 只打一轮在本项目 E=1 的冻结训练壳下过于脆弱
3. 末段 5 轮窗口既满足“模型已接近收敛后植入后门”，也能减少单轮偶然波动

冻结窗口如下：

| 数据集 | 总轮数 | 攻击轮次 |
|:------|:------|:---------|
| CIFAR-10 | 100 | [95, 96, 97, 98, 99] |
| MNIST | 50 | [45, 46, 47, 48, 49] |

5 轮 smoke test 使用短窗口：

| smoke 配置 | 攻击轮次 |
|:-----------|:---------|
| 5 轮 smoke | [3, 4] |

---

## 4. 实现方案

### 4.1 攻击路径划分

Scaling Attack 在本仓库中必须拆成两条路径同时实现：

1. 客户端路径：后门训练
2. 服务端路径：模型缩放

攻击仍然保持 `attack_type=model_replacement`。不要把后门训练塞进 LF 的 data poisoning 路径，也不要改动 ASR 评估逻辑。

### 4.2 客户端侧实现

客户端后门训练必须放在 trainer 训练路径中完成，最终位置冻结为：

- trainer 文件：[python/examples/federate/prebuilt_jobs/shieldfl/trainer/verifl_trainer.py](python/examples/federate/prebuilt_jobs/shieldfl/trainer/verifl_trainer.py)

实现要求：

1. 通过 `self.id < byzantine_client_num` 判断当前客户端是否恶意
2. 恶意客户端需额外判断当前轮次是否在 `attack_training_rounds` 中；非攻击轮次的恶意客户端走正常训练路径（不注入 trigger，与良性客户端完全一致）
3. 仅在攻击轮次，恶意客户端在每个训练 batch 中固定选择 20 张样本改造成后门样本
4. 若 batch 实际大小小于 20，则注入数 = batch 实际大小
5. 良性客户端 batch 完全不改
6. 训练使用的 trigger 注入语句必须与 ASR 评估完全一致：

```python
images[:, :, -trigger_size:, -trigger_size:] = trigger_value
```

7. 被选中的后门样本标签必须统一改为 `target_label`
8. 后门样本选择必须使用隔离 RNG，禁止污染全局 `random` / `np.random`

> **设计说明**：Bagdasaryan 原文及 FLTrust 等后续工作使用数据扩增方式（复制 p 比例样本 → 添加 trigger → 追加到训练集）。本项目改用 batch 内原位替换（固定 20 张/batch），原因是：(a) 避免动态修改 DataLoader 的复杂度；(b) 每 batch 20/64 ≈ 31% 的注入比例与社区常见 PDR 在同一数量级。此简化不改变攻击核心语义（trigger 注入 + 缩放放大），且训练 / 评估 trigger 一致，结果可比。

推荐实现规则：

- 为每个恶意客户端创建独立 RNG：`np.random.default_rng(random_seed + client_id + 2**20)`
- 每个 batch 基于该 RNG 从 batch 下标中无放回抽取 20 个位置
- 每轮训练开始时打印该客户端的 RNG seed 和 backdoor_per_batch

### 4.3 服务端侧实现

服务端缩放逻辑冻结在：

- 攻击类文件：[python/fedml/core/security/attack/model_replacement_backdoor_attack.py](python/fedml/core/security/attack/model_replacement_backdoor_attack.py)

必须实现以下规则：

1. `attack_training_rounds` 作为唯一合法配置名
2. 攻击轮外直接返回原始列表，不做任何修改
3. 恶意客户端集合固定为 `[0, 1, 2]`
4. 在攻击轮次，对 `raw_client_grad_list[0]`、`[1]`、`[2]` 分别执行缩放
5. 禁止使用 `random.randrange()` 选择恶意客户端
6. 禁止 `pop + insert` 的原地删插操作
7. 必须保留列表长度与非恶意客户端条目的原始顺序

正确的更新规则是对恶意下标直接原位替换：

```python
raw_client_grad_list[idx] = (num, modified_model)
```

### 4.4 BN 参数处理

缩放覆盖范围冻结为：

- 需要缩放：所有 weight、bias、running_mean、running_var
- 不需要缩放：num_batches_tracked

也就是说，本项目对 BN 参数的约束是：

1. running_mean 必须参与缩放
2. running_var 必须参与缩放
3. num_batches_tracked 必须跳过

当前代码中 `is_weight_param(k)`（位于 [python/fedml/core/security/common/utils.py](python/fedml/core/security/common/utils.py)）将 running_mean 和 running_var 同时排除，**不满足**上述要求。缩放时必须替换为以下逻辑：

```python
def should_scale_param(k):
    """num_batches_tracked 不参与缩放，其余参数（含 running_mean/running_var）全部参与。"""
    return "num_batches_tracked" not in k
```

缩放循环中将 `if is_weight_param(k):` 改为 `if should_scale_param(k):`。`is_weight_param` 函数本身保持不变（仅影响 `vectorize_weight` 等工具函数，不在此次修改范围）。

### 4.5 攻击轮次与日志

以下日志是强制要求，不可省略：

1. server 启动时：

```text
Scaling attack init | malicious_client_ids=[0,1,2] | gamma=10 | attack_rounds=[...]
```

2. client 启动时：

```text
Scaling backdoor init | client_id=0 | is_malicious=True | backdoor_per_batch=20 | target_label=0 | trigger_size=3 | trigger_value=1.0
```

3. 每个恶意客户端每轮训练结束时：

```text
Scaling backdoor epoch summary | client_id=0 | round=95 | poisoned_samples=XXX | poisoned_batches=YYY
```

4. 服务端每个攻击轮：

```text
Scaling apply | round=95 | malicious_idx=0 | gamma=10
Scaling apply | round=95 | malicious_idx=1 | gamma=10
Scaling apply | round=95 | malicious_idx=2 | gamma=10
```

这些日志是验收的一部分。

---

## 5. YAML 最终字段

最终必须支持的 YAML 字段如下：

```yaml
enable_attack: true
attack_type: "model_replacement"
eval_asr: true

byzantine_client_num: 3
scale_gamma: 10
attack_training_rounds: [95, 96, 97, 98, 99]

target_label: 0
trigger_size: 3
trigger_value: 1.0
backdoor_per_batch: 20

attacker_epochs: null
attacker_lr: null
attacker_weight_decay: null
attacker_noise_sigma: 0
```

当前脚本 [python/examples/federate/prebuilt_jobs/shieldfl/scripts/run_experiment.sh](python/examples/federate/prebuilt_jobs/shieldfl/scripts/run_experiment.sh) 中，以下行为必须同步落地：

1. `--attack model_replacement` 自动设置 `eval_asr: true`
2. 自动写入 `byzantine_client_num=3`
3. 自动写入 `scale_gamma=10`
4. 根据数据集自动写入 `attack_training_rounds`
5. 写入 `backdoor_per_batch=20`
6. 不再写入 `poisoned_training_round`
7. 不再把“两个属性名同时设置”作为临时兼容方案8. 当 `attack_type=model_replacement` 时，**不写入** `attack_mode`（该字段仅用于 `label_flipping`；当前脚本对所有攻击硬编码 `attack_mode: "flip"`，会与 model_replacement 路径冲突，必须条件跳过）
---

## 6. 不改动的部分

以下组件不在本次实现范围内：

| 组件 | 原因 |
|:-----|:-----|
| [python/examples/federate/prebuilt_jobs/shieldfl/eval/asr.py](python/examples/federate/prebuilt_jobs/shieldfl/eval/asr.py) | 已有 trigger 注入规则正确，只允许复用，不允许改语义 |
| [python/examples/federate/prebuilt_jobs/shieldfl/data/data_loader.py](python/examples/federate/prebuilt_jobs/shieldfl/data/data_loader.py) | 不在数据加载阶段做永久污染 |
| VeriFL / Baseline 聚合主干 | 只在 on_before_aggregation 攻击钩子注入 scaling |
| M1.5 clean baseline | 已冻结，不重跑 |

---

## 7. 实验矩阵

### 7.1 Smoke test

Smoke test 只验证链路，不作为正式结论数据。

命令冻结为：

```bash
bash scripts/run_experiment.sh \
  --model ResNet18 --dataset cifar10 \
  --attack model_replacement --defense none --aggregator fedavg \
  --pmr 0.3 --alpha 0.5 --seed 0 \
  --rounds 5 --clients 10 --epochs 1 --batch_size 64 --gpu
```

该 smoke test 必须自动写入：

- `attack_training_rounds=[3,4]`
- `byzantine_client_num=3`
- `scale_gamma=10`
- `backdoor_per_batch=20`

### 7.2 正式实验

正式实验矩阵冻结为：

| 数据集 | alpha | seed | 轮数 | 模型 | 数量 |
|:------|:------|:-----|:-----|:-----|:-----|
| CIFAR-10 | 0.1, 0.3, 0.5, 100 | 0, 1, 2 | 100 | ResNet18 | 12 |
| MNIST | 0.1, 0.3, 0.5, 100 | 0, 1, 2 | 50 | LeNet5 | 12 |

合计 24 组。

所有正式实验都必须继承以下固定项：

- clients=10
- client_num_per_round=10
- pmr=0.3
- byzantine_client_num=3
- scale_gamma=10
- backdoor_per_batch=20
- target_label=0
- trigger_size=3
- trigger_value=1.0
- attacker_epochs=null
- attacker_lr=null
- attacker_weight_decay=null

### 7.3 必做控制实验

为证明“是缩放导致了后门进入全局模型，而不是单纯后门训练本身”，必须额外做 3 组控制实验：

| 数据集 | alpha | seed | gamma |
|:------|:------|:-----|:------|
| CIFAR-10 | 0.5 | 0, 1, 2 | 1 |

其余参数与正式实验完全相同。

这 3 组不计入正式 24 组，但属于必做验收实验。

---

## 8. 指标定义

### 8.1 Clean Accuracy

使用 `metrics.jsonl` 中每轮 `test_accuracy`。

正式验收取最终轮：

- CIFAR-10 取 round=99
- MNIST 取 round=49

### 8.2 ASR

使用 `metrics.jsonl` 中每轮 `asr`。

正式验收同样取最终轮：

- CIFAR-10 取 round=99 的 ASR
- MNIST 取 round=49 的 ASR

### 8.3 Clean Accuracy Drop

与对应的 M1.5 clean baseline 做最终轮对比，定义为：

$$\text{Clean Drop (pp)} = \text{Acc}_{\text{baseline}} - \text{Acc}_{\text{scaling}}$$

单位是百分点，不用相对百分比。

### 8.4 Seed 统计

每个 alpha 配置报告：

1. seed=0,1,2 三个最终轮数值
2. mean ± std
3. 2/3 seed 是否满足阈值

---

## 9. 验收标准

验收分三层。Layer 1 和 Layer 2 全部通过后，才允许进入 Layer 3。Layer 3 中分为“阻塞验收项”和“结果记录项”。

### 9.1 Layer 1：代码正确性

#### AC-1 攻击轮次配置生效

使用 5 轮 smoke test 配置，但**将 attack_training_rounds 覆盖为 [3]**（而非默认 smoke 的 [3,4]），以精确测试轮次过滤逻辑。

通过条件：

1. server 日志仅 round=3 出现 `Scaling apply`
2. round=0,1,2,4 均不得出现 `Scaling apply`
3. 恶意客户端日志仅 round=3 出现后门注入（`poisoned_in_batch>0`），其余轮恶意客户端的注入日志为 0 或不打印
4. 不得触发属性名错误或回退到"每轮攻击"

#### AC-2 恶意客户端 batch 注入数量正确

在 smoke test 中检查 client_id=0 的前 3 个 batch 日志。

通过条件：

1. 每个 batch 记录 `poisoned_in_batch=20`
2. 若最后一个 batch 不足 20 条，则记录值等于该 batch 实际大小
3. 良性客户端日志中 `poisoned_in_batch` 必须恒为 0 或不打印该字段

#### AC-3 训练 trigger 与评估 trigger 完全一致

通过条件：

1. 训练代码中使用与 [python/examples/federate/prebuilt_jobs/shieldfl/eval/asr.py](python/examples/federate/prebuilt_jobs/shieldfl/eval/asr.py) 相同的 patch 位置表达式
2. 单元测试中，对同一张图像分别走“训练时注入函数”和“ASR 注入函数”，patched tensor 的最大绝对差必须等于 0

#### AC-4 恶意客户端集合固定且一致

通过条件：

1. client 日志打印的恶意客户端集合固定为 `[0,1,2]`
2. server 日志打印的恶意客户端集合固定为 `[0,1,2]`
3. 同一配置重复运行两次，上述集合完全一致

#### AC-5 gamma 可配且数值正确

通过条件：

1. 配置 `scale_gamma=10` 时，攻击轮日志中必须打印 `gamma=10`
2. 配置 `scale_gamma=1` 时，攻击轮日志中必须打印 `gamma=1`
3. 不允许仍然隐式使用 `participant_num` 而不打印最终 gamma

#### AC-6 BN 参数缩放范围正确

通过条件：

1. 单元测试中，恶意模型的 `running_mean` 在缩放后数值发生变化
2. 单元测试中，恶意模型的 `running_var` 在缩放后数值发生变化
3. 单元测试中，`num_batches_tracked` 保持整数且不参与缩放

#### AC-7 非恶意条目不被破坏

通过条件：

1. `raw_client_grad_list` 的长度在攻击前后相同
2. 非恶意下标 3..9 的 sample_num 不变
3. 非恶意下标 3..9 的模型参数逐 key 与攻击前完全相同

### 9.2 Layer 2：链路正确性

#### AC-8 5 轮 smoke test 跑通

通过条件：

1. 命令 exit code = 0
2. 生成 metrics 文件
3. round 0..4 均有 `test_accuracy`、`test_loss`、`asr`
4. `attack_type` 字段值为 `model_replacement`

#### AC-9 smoke test 中 ASR 非零且明显高于随机猜测

通过条件：

1. round=4 的 ASR ≥ 0.20
2. round=4 的 ASR 不能为 null，也不能恒等于 0

说明：CIFAR-10 / MNIST 均为 10 类分类，随机猜测命中 target_label 的概率约为 0.10。0.20 是链路有效性的最低门槛（显著高于随机），不是正式实验门槛。

#### AC-10 客户端与服务端恶意 ID 对齐

通过条件：

1. smoke test 中 client_id=0,1,2 均打印恶意训练日志
2. 同一轮 server 仅对 malicious_idx=0,1,2 打印缩放日志
3. client_id=3..9 不得打印恶意训练日志，server 也不得对这些 idx 缩放

### 9.3 Layer 3：正式实验验收

#### 阻塞验收项

#### AC-11 CIFAR-10 攻击有效性

在 12 组 CIFAR-10 正式实验中，按 alpha 聚合 3 个 seed 的最终轮 ASR。

通过条件：

1. 至少 3 个 alpha 配置满足 `mean(final ASR) ≥ 0.80`
2. 对每个满足条件的 alpha，3 个 seed 中至少 2 个 seed 满足 `final ASR ≥ 0.80`

#### AC-12 MNIST 攻击有效性

在 12 组 MNIST 正式实验中，按 alpha 聚合 3 个 seed 的最终轮 ASR。

通过条件：

1. 至少 3 个 alpha 配置满足 `mean(final ASR) ≥ 0.90`
2. 对每个满足条件的 alpha，3 个 seed 中至少 2 个 seed 满足 `final ASR ≥ 0.90`

#### AC-13 缩放的因果性成立

使用 3 组 `gamma=1` 控制实验，与对应的 `gamma=10` 正式实验比较（CIFAR-10, alpha=0.5, seed=0/1/2）。

通过条件：

1. `mean(final ASR, gamma=10) - mean(final ASR, gamma=1) ≥ 0.30`
2. 三个 seed 中至少 2 个 seed 满足 `final ASR(gamma=10) > final ASR(gamma=1)`

如果 AC-13 失败，则说明“后门训练在起作用，但 scaling 本身未被证明是必要条件”，本次复现不得结项。

#### 结果记录项

#### AC-14 CIFAR-10 clean task 保持度

对 12 组 CIFAR-10 正式实验，计算最终轮 clean accuracy 相对 M1.5 baseline 的 drop。

记录标准：

1. 若至少 3 个 alpha 配置满足 `mean(clean drop) ≤ 5.0 pp`，记为 `PASS`
2. 否则记为 `RECORD_ONLY`

`RECORD_ONLY` 不阻塞实现验收，但必须在最终实验报告中单独说明“攻击有效但 clean task 代价偏高”。

#### AC-15 MNIST clean task 保持度

对 12 组 MNIST 正式实验，计算最终轮 clean accuracy 相对 M1.5 baseline 的 drop。

记录标准：

1. 若至少 3 个 alpha 配置满足 `mean(clean drop) ≤ 3.0 pp`，记为 `PASS`
2. 否则记为 `RECORD_ONLY`

---

## 10. 必交付物

工程完成时，必须同时交付以下内容：

1. 代码修改
2. 单元测试文件 `test_scaling_correctness.py`
3. smoke 脚本 `run_m2_scaling_smoke.sh`
4. 24 组正式实验 metrics.jsonl
5. 3 组 `gamma=1` 控制实验 metrics.jsonl
6. 实验报告 `M2_SCALING_EXPERIMENT_REPORT.md`

实验报告中必须包含：

1. 24 组正式实验的最终轮 clean accuracy 与 final ASR 总表
2. 3 组 gamma=1 控制实验对照表
3. AC-11 至 AC-15 的逐项判定
4. client/server 恶意 ID 一致性日志截图或文本摘录

---

## 11. 结项判定规则

满足以下条件时，Scaling Attack 复现可判定完成：

1. AC-1 至 AC-13 全部通过
2. 24 组正式实验 + 3 组控制实验全部完成
3. 交付物 1 至 6 全部齐全

AC-14 与 AC-15 用于记录攻击的 clean-task 代价，不作为阻塞项。

如果 AC-11、AC-12、AC-13 中任一失败，则本次复现不能结项。
