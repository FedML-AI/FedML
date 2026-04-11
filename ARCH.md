# FedML 项目整体模块构成梳理（基于 docs 关键词定位 + 仓库代码映射）

---

## 1. 一句话总览

FedML 不是单一的“联邦学习算法仓库”，而是一个围绕 **训练（Train）/ 部署（Deploy）/ 调度（Launch）/ 联邦学习（Federate）/ MLOps / 多端 SDK** 构建的完整 AI 基础设施仓库。  
其中：

- **联邦学习主线**集中在 `python/fedml/{cross_device,cross_silo,simulation,fa,core}` 及 `python/examples/federate/*`
- **基线测试 / benchmark 主线**集中在 `python/examples/federate/prebuilt_jobs/*`，尤其是 `fedcv`、`fednlp`、`fedgraphnn`、`healthcare`
- **攻击模拟 / 安全防御主线**集中在 `python/fedml/core/security/*`、`python/fedml/core/dp/*`、`python/fedml/{cross_silo/secagg,cross_silo/lightsecagg}`、`python/examples/federate/{security,privacy}`

---

## 2. 从文档看，FedML 的核心产品/能力分层

根据文档 index.md 与 `docs/federate/index.md`，FedML（文档中也以 TensorOpera Open Source/Federate 形式出现）主要由以下能力层构成：

### 2.1 Launch：任务启动与调度
对应代码：
- scheduler

作用：
- 统一启动与调度 AI 作业
- 可对接 GPU 云、私有集群、边缘节点等资源

### 2.2 Deploy：模型部署与服务
对应代码：
- serving
- scalellm

作用：
- 模型服务化
- LLM 低显存部署/扩展

### 2.3 Train：训练与可观测性
对应代码：
- train
- `python/fedml/train/cross_cloud`
- llm
- mlops

作用：
- 分布式训练
- LLM 训练/微调
- 实验跟踪与可观测性

### 2.4 Federate：联邦学习与联邦分析
对应代码：
- cross_device
- cross_silo
- simulation
- fa

作用：
- 跨设备联邦学习
- 跨机构/跨组织联邦学习
- FL 模拟器
- 联邦分析（Federated Analytics）

### 2.5 Core / API / CLI / Data / Model
对应代码：
- core
- api
- cli
- data
- model
- ml
- utils

作用：
- 提供底层通信、聚合、安全、隐私、设计模式、公共接口、预置数据与模型等基础设施

---

## 3. 从仓库结构看，FedML 顶层由哪些模块构成

从当前工作区实际目录看，FedML 项目不仅有 Python 主库，还包含多端 SDK、运维与安装体系。

### 3.1 顶层模块总览

| 顶层目录 | 作用概述 | 与 FL / 基线 / 攻击 的关系 |
|---|---|---|
| python | 主 Python 库与核心实现 | **最核心** |
| android | Android 端 SDK / Demo | 与**跨设备联邦学习**直接相关 |
| ios | iOS 端支持 | 与**跨设备联邦学习**相关 |
| iot | IoT 场景与样例 | 与**联邦学习、攻击检测、IoT 安全**相关 |
| `data/` | 数据准备与示例数据 | 为训练/联邦实验提供支撑 |
| examples | 顶层示例集合 | 入口性质，核心示例更多在 examples |
| installation | 各平台安装、容器、K8s 安装 | 工程部署支撑 |
| devops | Docker、K8s、CI/CD、脚本 | 工程化支撑 |
| doc.fedml.ai | 文档站源码（你要求我重点参考的部分） | 用来定位官方模块说明 |
| research | 研究资料与论文索引 | 理论背景与 benchmark 论文线索 |
| README.md 等 | 项目介绍、贡献说明 | 总览性质 |

### 3.2 结论

如果把整个仓库看成一座楼：

- python 是主承重结构
- android、ios、iot 是通向真实设备世界的门
- devops、installation 是施工队和后勤
- doc.fedml.ai 是导览图
- research 是论文陈列馆

---

## 4. Python 主库的核心模块构成

根据 README.md 与 fedml 实际目录，fedml 是整个仓库最重要的实现主体。

### 4.1 fedml 主要子模块

| 模块 | 代码位置 | 作用 |
|---|---|---|
| API | api | 对外公共 Python API |
| CLI | cli | 命令行工具实现 |
| Core | core | 底层通信、聚合、安全、DP、流程抽象等 |
| Computing/Scheduler | computing | 调度、设备管理、运行编排 |
| Cross-cloud | cross_cloud | 跨云训练 |
| Cross-device | cross_device | 跨设备 FL（手机/IoT） |
| Cross-silo | cross_silo | 跨机构/跨组织 FL |
| Simulation | simulation | FL 仿真器 |
| FA | fa | 联邦分析 |
| Train | train | 分布式训练与 LLM 训练 |
| Serving | serving | 模型部署与服务 |
| ScaleLLM | scalellm | LLM 推理/扩展 |
| Data | data | 数据集与数据加载 |
| Model | model | 模型仓库 |
| ML | ml | 各 ML 框架适配与聚合器/训练器等 |
| MLOps | mlops | 实验追踪、平台能力 |
| Device | device | 设备与资源管理 |
| Distributed | distributed | 分布式训练相关 |
| Utils | utils | 工具函数 |

### 4.2 `fedml/core` 是真正的“底盘”

core 下实际包含：

- `alg_frame/`
- `common/`
- `contribution/`
- `data/`
- `distributed/`
- `dp/`
- `fhe/`
- `mlops/`
- `mpc/`
- `schedule/`
- `security/`

这说明 `core` 不是单纯“工具包”，而是以下能力的统一底座：

- 联邦训练抽象流程
- 通信后端与分布式执行
- 差分隐私
- 安全攻击与防御
- 同态加密 / MPC
- 贡献评估
- MLOps 支撑

文档中说 “all algorithms and scenarios are built based on the core package”，这点和代码结构完全一致。

---

## 5. 哪些模块与联邦学习直接相关

这一部分是你最关心的主线之一。

## 5.1 联邦学习主干模块

### cross_device
作用：
- 面向手机、IoT、边缘终端的跨设备联邦学习
- 文档明确提到 Android/iOS/embedded Linux 场景

关联顶层目录：
- android
- ios
- iot

### cross_silo
作用：
- 跨组织、跨账号、跨数据孤岛的联邦学习
- 适合企业/机构/边缘服务器

该目录下还包含安全聚合相关实现：
- `cross_silo/secagg`
- `cross_silo/lightsecagg`

### simulation
作用：
- FL 仿真器
- 文档明确支持：
  - 单进程模拟
  - MPI-based 模拟
  - NCCL-based 模拟

这是联邦学习研究与算法验证的关键模块。

### fa
作用：
- Federated Analytics（联邦分析）
- 不是标准模型训练，但属于联邦范式的重要分支

---

## 5.2 联邦学习的示例与应用层目录

对应目录：
- cross_silo
- cross_device
- simulation
- federated_analytics
- quick_start
- flow

这些目录的意义是：

- 给出不同通信后端的运行方式
- 展示横向 FL、分层 FL、跨设备 FL、模拟 FL
- 给研究人员/工程人员提供可直接改造的模板

例如 `cross_silo` 下可见：
- `mpi_fedavg_mnist_lr_example`
- `mqtt_s3_fedavg_mnist_lr_example`
- `light_sec_agg_example`
- `mqtt_s3_fedavg_attack_mnist_lr_example`
- `mqtt_s3_fedavg_fhe_mnist_lr_example`

也就是说，联邦学习不是“单一路径”，而是通信后端、隐私增强、攻击模拟、分层架构都能组合。

---

## 5.3 Spotlight 项目中的联邦学习扩展

目录：
- fedllm
- unitedllm

其中与联邦学习更直接相关的是：

### fedllm
作用：
- Federated Learning on LLMs
- 是 FedML 向“大模型联邦学习”扩展的重要方向

### unitedllm
作用：
- 偏跨云分布式 LLM 训练
- 更偏 Train / Cross-cloud，而非纯 FL

---

## 5.4 多端 FL 相关支撑模块

除了 Python 主线，以下目录也与联邦学习落地密切相关：

### android
- Android SDK、应用、demo
- 对应文档中 cross-device smartphone FL 能力

### ios
- iOS 端联邦/边缘能力

### iot
- IoT 场景
- 其中 `anomaly_detection_for_cybersecurity/` 特别值得关注，和联邦学习在安全场景的应用强相关

---

## 6. 哪些模块与基线测试 / benchmark 有关

这部分在 FedML 中非常重要，而且不是“附带品”，而是官方文档中明确强调的研究与评测能力。

## 6.1 最核心的 benchmark 目录：prebuilt_jobs

该目录实际包含：

- `fedcv/`
- `fedgraphnn/`
- `fedllm/`
- `fednlp/`
- `healthcare/`

文档 `docs/federate/examples.md` 明确指出，这些是 Federate 的 **Pre-built Jobs**，用于支撑联邦学习研究与产品化。

---

## 6.2 具体 benchmark / baseline 模块

### 6.2.1 `fednlp/`
作用：
- NLP 场景联邦学习 benchmark
- 文档中称：**Benchmarking Federated Learning Methods for Natural Language Processing Tasks**

覆盖任务：
- text classification
- sequence tagging
- question answering
- seq2seq

特点：
- 统一 Transformer / FL 接口
- 比较多种 FL 方法
- 强调 non-IID 划分下的评测

### 6.2.2 `fedcv/`
作用：
- 计算机视觉联邦学习 benchmark
- 文档中称：**A Federated Learning Framework for Diverse Computer Vision Tasks**

覆盖任务：
- image classification
- image segmentation
- object detection

特点：
- 提供非 IID benchmark 数据、模型和参考 FL 算法
- 是非常典型的“研究 benchmark 框架”

### 6.2.3 `fedgraphnn/`
作用：
- 图神经网络联邦学习 benchmark
- 文档中称：**A Federated Learning Benchmark System for Graph Neural Networks**

特点：
- 支持多种图数据集、GNN 模型、FL 算法
- 专门服务于 federated GNN 研究

### 6.2.4 `healthcare/`
作用：
- 医疗场景 cross-silo FL benchmark
- 该目录 README 明确基于 **FLamby**：
  - “Datasets and Benchmarks for Cross-Silo Federated Learning in Realistic Healthcare Settings”

特点：
- 真实医疗联邦 benchmark
- 目录中可见 `fed_ixi`、`fed_isic2019`、`fed_kits19`
- 代码里多处直接引用 FLamby 的 `Baseline` 与 `BaselineLoss`

### 6.2.5 `fedllm/`
作用：
- 大模型联邦学习任务
- 更偏新方向应用/实验平台

---

## 6.3 与 benchmark 强相关但不是主 benchmark 框架的目录

### centralized
作用：
- 集中式训练示例
- README.md 明确写着：
  - “Some centralized trainer code examples for benchmarking purposes.”

这意味着它常被用于：
- 与联邦学习效果对比
- 作为集中式 baseline

### grpc_benchmark
作用：
- 系统/通信层 benchmark
- 关注 gRPC 通信性能，而不是 FL 算法效果本身

### `doc.fedml.ai/docs/federate/simulation/benchmark/*`
作用：
- FL 模拟 benchmark 文档与实验说明
- 更偏“实验说明/性能评测文档”

---

## 6.4 基线测试相关模块总结

如果你问“FedML 哪些模块最适合做 baseline / benchmark？”  
答案是：

**第一梯队：**
- fednlp
- fedcv
- fedgraphnn
- healthcare

**第二梯队：**
- centralized
- simulation
- grpc_benchmark

---

## 7. 哪些模块与攻击模拟、安全防御、隐私保护有关

这部分是 FedML 的另一条很强的主线，而且实现是“嵌入式”的——不是孤零零的一堆 demo。

## 7.1 安全核心：security

这是攻击模拟与防御机制的核心目录，实际包含：

- `fedml_attacker.py`
- `fedml_defender.py`
- `attack/`
- `defense/`
- `common/`
- constants.py

文档 readme.md 与 overview.md 都明确表明：

- `FedMLAttacker`：模拟 FL 中的攻击
- `FedMLDefender`：注入防御逻辑
- 整体被称为 **FedMLSecurity**
- 它本质上是一个 **attacks & defenses benchmark**

---

## 7.2 攻击模块（Attack）

attack 下实际有：

- `byzantine_attack.py`
- `dlg_attack.py`
- `invert_gradient_attack.py`
- `label_flipping_attack.py`
- `backdoor_attack.py`
- `edge_case_backdoor_attack.py`
- `model_replacement_backdoor_attack.py`
- `revealing_labels_from_gradients_attack.py`

从文档与代码可以归纳出支持的攻击类型：

### 7.2.1 Byzantine 攻击
- random
- zero
- flip

### 7.2.2 梯度泄露 / 数据重建攻击
- DLGAttack
- InvertAttack
- RevealingLabelsFromGradientsAttack

### 7.2.3 数据投毒 / 标签翻转
- LabelFlippingAttack

### 7.2.4 后门攻击
- BackdoorAttack
- EdgeCaseBackdoorAttack
- ModelReplacementBackdoorAttack

这里已经不是“有没有安全研究接口”的问题，而是**攻击菜单都快能点套餐了**。

---

## 7.3 防御模块（Defense）

defense 下实际有：

- `bulyan_defense.py`
- `cclip_defense.py`
- `coordinate_wise_median_defense.py`
- `coordinate_wise_trimmed_mean_defense.py`
- `crfl_defense.py`
- `foolsgold_defense.py`
- `geometric_median_defense.py`
- `krum_defense.py`
- `norm_diff_clipping_defense.py`
- `robust_learning_rate_defense.py`
- `slsgd_defense.py`
- `soteria_defense.py`
- `wbc_defense.py`
- `RFA_defense.py`
- 等

也就是说，FedML 不只是支持“攻击案例”，还支持大量主流鲁棒聚合与防御算法的对照实验。

---

## 7.4 隐私与密码学模块

### dp
作用：
- 差分隐私（DP）
- 包含：
  - fedml_differential_privacy.py
  - `mechanisms/`
  - `budget_accountant/`
  - `frames/`

### mpc
作用：
- 多方安全计算（MPC）
- 支撑 secure aggregation 等能力

### fhe
作用：
- 全同态加密相关聚合能力

### secagg
作用：
- secure aggregation 实现

### lightsecagg
作用：
- lightweight secure aggregation

这些模块说明，FedML 的“安全”不止是攻击模拟，还包括：

- 差分隐私
- 安全聚合
- MPC
- 同态加密

---

## 7.5 安全/隐私示例目录

### security
这是最直接的攻击/防御实验目录，包含：

- `mpi_fedavg_byzantine_krum_mnist_lr_example`
- `mqtt_s3_fedavg_attack_mnist_lr_example`
- `mqtt_s3_fedavg_attack_defense_cifar10_resnet56_example`
- `mqtt_s3_fedavg_byzantine_krum_mnist_lr_example`
- `mqtt_s3_fedavg_defense_mnist_lr_example`
- `fedMLSecurity_experiments`

特点：
- 配置里直接出现 `enable_attack: true`
- 配置里直接出现 `enable_defense: true`
- 是官方推荐的 FedMLSecurity 实验入口

### privacy
包含：
- `mpi_fedavg_dp_mnist_lr_example`
- `mqtt_s3_fedavg_cdp_mnist_lr_example`
- `mqtt_s3_fedavg_ldp_mnist_lr_example`

作用：
- 展示 DP 相关实验
- 对应集中式/本地/全局差分隐私变体

---

## 7.6 攻击模拟所依赖的数据与案例

### edge_case_examples
这个目录非常关键。

文档 `docs/federate/datasets-and-models.md` 与代码 README.md 都明确提到：

- `edge_case_examples`
- 对应论文：**Attack of the Tails: Yes, You Really Can Backdoor Federated Learning**

该目录下有：
- data_loader.py
- datasets.py
- README.md
- get_data.sh

也就是说，FedML 为某些攻击研究不只是给了算法接口，还给了专门的数据构造支持。

---

## 7.7 安全逻辑已嵌入 FL 主流程，而不是外挂

这一点非常重要。  
从 server_aggregator.py 可以看到：

- 初始化时会调用：
  - `FedMLAttacker.get_instance().init(args)`
  - `FedMLDefender.get_instance().init(args)`
  - `FedMLDifferentialPrivacy.get_instance().init(args)`

在聚合前会：
- 执行全局裁剪
- 执行数据重建攻击
- 执行模型攻击
- 执行防御前处理

在聚合时会：
- 走防御版聚合逻辑
- 或普通聚合逻辑

聚合后还会：
- 添加全局 DP 噪声
- 执行防御后处理

这说明 FedML 的安全/攻击/隐私机制不是松散脚本，而是已经和联邦训练主循环集成。

同样，在：
- client_trainer.py
- FedAvgAPI.py
- `python/fedml/simulation/mpi/async_fedavg/*`
- `python/fedml/simulation/mpi/fednova/*`

中也都能看到攻击器/防御器/DP 的接入。

---

## 8. 哪些模块同时与“联邦学习 + 基线测试 + 攻击模拟”三者交叉有关

这是最值得关注的“交叉地带”。

| 模块 | 联邦学习 | 基线/Benchmark | 攻击模拟/安全 |
|---|---:|---:|---:|
| core | 是 | 间接支撑 | **是** |
| simulation | **是** | **是** | **是** |
| cross_silo | **是** | 是 | **是** |
| prebuilt_jobs | **是** | **是** | 部分相关 |
| security | **是** | **是** | **是** |
| privacy | **是** | 是 | **是** |
| edge_case_examples | 否（支撑） | 是 | **是** |
| anomaly_detection_for_cybersecurity | **是** | 是 | **强相关** |

其中最典型的三类交叉核心是：

### 8.1 simulation
为什么重要：
- 研究新算法通常先在 simulation 做
- 安全攻击/防御也常先在 simulation 路径验证
- benchmark 实验也最容易在 simulation 路径复现

### 8.2 security
为什么重要：
- 既是 FL 实验
- 又是 attack/defense benchmark
- 还是“可跑”的官方配置入口

### 8.3 prebuilt_jobs
为什么重要：
- 它是 benchmark 框架和应用场景的落地层
- 能把联邦学习从“算法”推进到 NLP/CV/GNN/Healthcare/IoT 具体任务

---

## 9. 如果按“研究者/开发者视角”来理解整个项目

可以把 FedML 理解为下面这几层：

### 第 1 层：底层联邦与安全内核
- core
- cross_silo
- cross_device
- simulation
- fa

### 第 2 层：数据、模型、聚合、训练抽象
- data
- model
- ml

### 第 3 层：安全与隐私增强
- security
- dp
- mpc
- fhe
- secagg
- lightsecagg

### 第 4 层：benchmark 与应用实验
- fednlp
- fedcv
- fedgraphnn
- healthcare
- security
- privacy

### 第 5 层：平台化、调度、部署、多端
- computing
- train
- serving
- android
- ios
- iot
- devops

---

## 10. 结论

## 10.1 FedML 由什么模块构成？

如果只抓主干，FedML 可概括为：

- **联邦学习模块**：`cross_device`、`cross_silo`、`simulation`、`fa`
- **训练模块**：`train`、`cross_cloud`
- **部署模块**：`serving`、`scalellm`
- **调度与设备模块**：`computing/scheduler`、`device`
- **底层基础设施模块**：`core`、`distributed`、`mlops`
- **数据与模型模块**：`data`、`model`、`ml`
- **接口层模块**：`api`、`cli`
- **示例与 benchmark 模块**：`python/examples/*`
- **多端与落地模块**：android、ios、iot
- **工程化与部署模块**：installation、devops
- **文档与研究模块**：doc.fedml.ai、research

## 10.2 哪些模块与联邦学习有关？

最核心的是：

- cross_device
- cross_silo
- simulation
- fa
- `python/examples/federate/*`
- fedllm
- android、ios、iot

## 10.3 哪些模块与基线测试 / benchmark 有关？

最核心的是：

- fednlp
- fedcv
- fedgraphnn
- healthcare
- centralized
- simulation
- grpc_benchmark

## 10.4 哪些模块与攻击模拟有关？

最核心的是：

- security
- attack
- defense
- dp
- mpc
- fhe
- secagg
- lightsecagg
- security
- privacy
- edge_case_examples

---

## 11. 给你的一个实用阅读顺序建议

如果你接下来要继续深入源码，我建议按这个顺序看：

1. index.md  
2. index.md  
3. README.md  
4. core  
5. `python/fedml/{cross_silo,simulation,cross_device}`  
6. prebuilt_jobs  
7. security  
8. security  
9. privacy  

这样可以先搭骨架，再看肌肉，最后看“防身术”。

---

## 已完成清单

- [x] 定位文档关键页面
- [x] 梳理主仓库模块结构
- [x] 识别联邦学习相关模块
- [x] 识别基线与攻击模块
- [x] 整理完整 Markdown 文档

## 补充一句

如果你愿意，我下一步可以继续把这份文档**再细化成一张“目录 → 功能 → 典型入口文件”的表**，或者专门再出一版：

- **FedML 联邦学习源码阅读地图**
- **FedML 安全攻击/防御源码阅读地图**
- **FedML benchmark 复现实验入口清单**

这三份会更适合真正下手读代码。