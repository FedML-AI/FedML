# Getting Started

Thank you for visiting! Here is a quick tour of FedML Open Source Library ([https://github.com/FedML-AI/FedML](https://github.com/FedML-AI/FedML)) and MLOps Platform ([https://open.fedml.ai](https://open.fedml.ai)) with examples for different scenarios. 

## **FedML Feature Overview**
![image](../../_static/image/4animals.png)

The FedML logo reflects the mission of FedML Inc. We aim to build simple and versatile APIs for machine learning running anywhere and at any scale.
In other words, FedML supports both federated learning for data silos and distributed training for acceleration with MLOps and Open Source support, covering both cutting-edge academia research and industrial grade use cases. 

- **FedML Cheetah** - Accelerate Model Training with User-friendly Distributed Training
- **FedML Parrot** - Simulating federated learning in the real world: (1) simulate FL using a single process (2) MPI-based FL Simulator (3) NCCL-based FL Simulator (fastest)
- **FedML Octopus** - Cross-silo Federated Learning for cross-organization/account training, including Python-based edge SDK.
- **FedML Beehive** - Cross-device Federated Learning for Smartphones and IoTs, including edge SDK for Android/iOS and embedded Linux.

- **FedML MLOps**: FedML's machine learning operation pipeline for AI running anywhere and at any scale.
- **Model Serving**: we focus on providing a better user experience for edge AI.

## **Quick Start for Open Source Library**
[https://github.com/FedML-AI/FedML](https://github.com/FedML-AI/FedML)

### Installation

To get started, let's first install FedML.

```Python
pip install fedml
```
For more installation methods, please refer to [installing FedML](./installation.md).

### A Quick Overview of the Code Architecture

In general, FedML source code architecture follows the paper which won the [Best Paper Award at NeurIPS 2020 (FL workshop)](https://chaoyanghe.com/wp-content/uploads/2021/02/NeurIPS-SpicyFL-2020-Baidu-best-paper-award-He-v2.pdf). Its original idea is presented at the live [video](https://www.youtube.com/watch?v=93SETZGZMyI) and 
[white paper](https://arxiv.org/abs/2007.13518) by FedML co-founder Dr. [Chaoyang He](https://chaoyanghe.com). 

![FedML Code Architecture](../_static/image/fedml.png)

As of March 2022, FedML has been made into an AI company which aims to provide machine learning capability from anywhere and at any scale. The python version of FedML [https://github.com/FedML-AI/FedML-refactor/tree/master/python](https://github.com/FedML-AI/FedML-refactor/tree/master/python) is now reorganized as follows:

**core**: The FedML low-level API package. This package implements distributed computing by communication backend such as MPI, NCCL, MQTT, gRPC, PyTorch RPC, and also supports topology management. 
Other low-level APIs related to security and privacy are also supported. All algorithms and Scenarios are built based on the "core" package.

**data**: FedML will provide some default datasets for users to get started. Customization templates are also provided.

**model**: FedML model zoo.

**device**: FedML computing resource management.

**simulation**: FedML parrot can support (1) simulating FL using a single process (2) MPI-based FL Simulator (3) NCCL-based FL Simulator (fastest)

**cross_silo**: Cross-silo Federated Learning for cross-organization/account training

**cross_device**: Cross-device Federated Learning for Smartphones and IoTs

**distributed**: Distributed Training: Accelerate Model Training with Lightweight Cheetah

**serve**: Model serving, tailored for edge inference

**mlops**: APIs related to machine learning operation platform (open.fedml.ai)

**centralized**: Some centralized trainer code examples for benchmarking purposes.

**utils**: Common utilities shared by other modules.

### Simplified APIs


<img src="../_static/image/apioverview.jpg" alt="drawing" style="width:60%;"/> 
<br />


Our philosophy behind our API design is to reduce the number of APIs to as few as possible while simultaneously maintaining the flexibility. 
The figure above shows the high-level overview of the API design. Essentially, each module has a package entry point (e.g., fedml.cross-silo) to manage related APIs, and FedML users can wrapper these APIs to meet their specific demands. 
Some high-level integrated APIs are also provided. For example, `fedml.run_simulation()` is just a one-line API for a quick start. 

Now let's get started with some simple examples. For Simplicity, FedML Parrot (simulator) supports one-line APIs as the following example:


```Python
# main.py

import fedml

if __name__ == "__main__":
    fedml.run_simulation()
```

```
python main.py
```

You will get the following output:
```
[FedML-Server(0) @device-id-0] [Sun, 01 May 2022 14:59:28] [INFO] [__init__.py:30:init] args = {'yaml_config_file': '', 'run_id': '0', 'rank': 0, 'yaml_paths': ['/Users/chaoyanghe/opt/anaconda3/envs/mnn37/lib/python3.7/site-packages/fedml-0.7.8-py3.7.egg/fedml/config/simulation_sp/fedml_config.yaml'], 'training_type': 'simulation', 'using_mlops': False, 'random_seed': 0, 'dataset': 'mnist', 'data_cache_dir': './data/mnist', 'partition_method': 'hetero', 'partition_alpha': 0.5, 'model': 'lr', 'federated_optimizer': 'FedAvg', 'client_id_list': '[]', 'client_num_in_total': 1000, 'client_num_per_round': 10, 'comm_round': 200, 'epochs': 1, 'batch_size': 10, 'client_optimizer': 'sgd', 'learning_rate': 0.03, 'weight_decay': 0.001, 'frequency_of_the_test': 5, 'using_gpu': False, 'gpu_id': 0, 'backend': 'single_process', 'log_file_dir': './log', 'enable_wandb': False}
[FedML-Server(0) @device-id-0] [Sun, 01 May 2022 14:59:28] [INFO] [device.py:14:get_device] device = cpu
[FedML-Server(0) @device-id-0] [Sun, 01 May 2022 14:59:28] [INFO] [data_loader.py:22:download_mnist] ./data/mnist/MNIST.zip
[FedML-Server(0) @device-id-0] [Sun, 01 May 2022 14:59:31] [INFO] [data_loader.py:57:load_synthetic_data] load_data. dataset_name = mnist
...
```
You can also customize the hyper-parameters with `fedml_config.yaml`. Check out [this tutorial for a one-line example](https://doc.fedml.ai/simulation/examples/sp_fedavg_mnist_lr_example.html) for details.

For flexibility, one-line API can also be expanded into five lines of APIs. To illustrate this, let's switch to FedML Octopus (cross-silo federated learning) as example (Source code: [https://github.com/FedML-AI/FedML/tree/master/python/examples/cross_silo/mqtt_s3_fedavg_mnist_lr_example](https://github.com/FedML-AI/FedML/tree/master/python/examples/cross_silo/mqtt_s3_fedavg_mnist_lr_example)).

In this example, the FL Client APIs are as follows:

```Python
import fedml
from fedml import FedMLRunner

if __name__ == "__main__":
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model
    model = fedml.model.create(args, output_dim)

    # start training
    FedMLRunner(args, device, dataset, model).run()

```

With these APIs, you only need to tune the hyper-parameters with the configuration file `fedml_config.yaml`. An example is as follows:
```yaml
common_args:
  training_type: "cross_silo"
  scenario: "horizontal"
  using_mlops: false
  random_seed: 0

environment_args:
  bootstrap: config/bootstrap.sh

data_args:
  dataset: "mnist"
  data_cache_dir: ~/fedml_data
  partition_method: "hetero"
  partition_alpha: 0.5

model_args:
  model: "lr"
  model_file_cache_folder: "./model_file_cache" # will be filled by the server automatically
  global_model_file_path: "./model_file_cache/global_model.pt"

train_args:
  federated_optimizer: "FedAvg"
  client_id_list:
  client_num_in_total: 1
  client_num_per_round: 2
  comm_round: 10
  epochs: 1
  batch_size: 10
  client_optimizer: sgd
  learning_rate: 0.03
  weight_decay: 0.001

validation_args:
  frequency_of_the_test: 1

device_args:
  worker_num: 2
  using_gpu: false
  gpu_mapping_file: config/gpu_mapping.yaml
  gpu_mapping_key: mapping_default

comm_args:
  backend: "MQTT_S3"
  mqtt_config_path: config/mqtt_config.yaml
  s3_config_path: config/s3_config.yaml
  # If you want to use your customized MQTT or s3 server as training backends, you should uncomment and set the following lines.
  #customized_training_mqtt_config: {'BROKER_HOST': 'your mqtt server address or domain name', 'MQTT_PWD': 'your mqtt password', 'BROKER_PORT': 1883, 'MQTT_KEEPALIVE': 180, 'MQTT_USER': 'your mqtt user'}
  #customized_training_s3_config: {'CN_S3_SAK': 'your s3 aws_secret_access_key', 'CN_REGION_NAME': 'your s3 region name', 'CN_S3_AKI': 'your s3 aws_access_key_id', 'BUCKET_NAME': 'your s3 bucket name'}

tracking_args:
   # When running on MLOps platform(open.fedml.ai), the default log path is at ~/fedml-client/fedml/logs/ and ~/fedml-server/fedml/logs/
  enable_wandb: false
```



Now let's run some examples to get a sense of how FedML simplifies federated learning in diverse real-world settings.
 
#### **FedML Parrot Examples**
Simulation with a Single Process (Standalone):

- [sp_fedavg_mnist_lr_example](./../simulation/examples/sp_fedavg_mnist_lr_example.md): 
  Simulating FL using a single process in your personal laptop or server. This is helpful for researchers hoping to try a quick algorithmic idea in small synthetic datasets (MNIST, shakespeare, etc.) and small models (ResNet-18, Logistic Regression, etc.). 

Simulation with Message Passing Interface (MPI):
- [mpi_torch_fedavg_mnist_lr_example](./../simulation/examples/mpi_torch_fedavg_mnist_lr_example.md): 
  MPI-based Federated Learning for cross-GPU/CPU servers.
  

Simulation with NCCL-based MPI (the fastest training):
- If your cross-GPU bandwidth is high (e.g., InfiniBand, NVLink, EFA, etc.), we suggest using this NCCL-based MPI FL simulator to accelerate your development. 

#### **FedML Octopus Examples**
Horizontal Federated Learning:

- [mqtt_s3_fedavg_mnist_lr_example](./../cross-silo/examples/mqtt_s3_fedavg_mnist_lr_example.md): an example to illustrate running horizontal federated learning in data silos (hospitals, banks, etc.)

Hierarchical Federated Learning:

- [hierarchical_fedavg_mnist_lr_example](./../cross-silo/examples/hierarchical_fedavg_mnist_lr_example.md): an example to illustrate running hierarchical federated learning in data silos (hospitals, banks, etc.). 
Here `hierarchical` implies that each FL Client (data silo) has multiple GPUs that can run local distributed training with PyTorch DDP, and the FL server then aggregates globally from the results received from all FL Clients.


#### **FedML Beehive Examples**

- [Federated Learning on Android Smartphones](./../cross-device/examples/mqtt_s3_fedavg_mnist_lr_example.md)



## **MLOps User Guide**
[https://open.fedml.ai](https://open.fedml.ai)

Currently, the project developed based on FedML Octopus (cross-silo) and Beehive (cross-device) can be smoothly deployed into the real-world system using FedML MLOps.

The FedML MLOps Platform simplifies the workflow of federated learning from anywhere and at any scale.
It enables zero-code, lightweight, cross-platform, and provably secure federated learning.
It enables machine learning from decentralized data at various users/silos/edge nodes, without the need to centralize any data to the cloud, hence providing maximum privacy and efficiency.

![image](../../_static/image/mlops_workflow.png)


The above figure shows the workflow, which is handled by a web UI that avoids using complex deployment. Check out the following live demo for details:

![image](../_static/image/mlops_invite.png)

3-Minute Introduction: [https://www.youtube.com/watch?v=E1k05jd1Tyw](https://www.youtube.com/watch?v=E1k05jd1Tyw)

Detailed guidance for the MLOps can be found at [FedML MLOps User Guide](./../mlops/user_guide.md). 

## **More Resources**


### References
```
@article{chaoyanghe2020fedml,
  Author = {He, Chaoyang and Li, Songze and So, Jinhyun and Zhang, Mi and Wang, Hongyi and Wang, Xiaoyang and Vepakomma, Praneeth and Singh, Abhishek and Qiu, Hang and Shen, Li and Zhao, Peilin and Kang, Yan and Liu, Yang and Raskar, Ramesh and Yang, Qiang and Annavaram, Murali and Avestimehr, Salman},
  Journal = {Advances in Neural Information Processing Systems, Best Paper Award at Federate Learning Workshop},
  Title = {FedML: A Research Library and Benchmark for Federated Machine Learning},
  Year = {2020}
}
```

### Ecosystem
<img src="../_static/image/started_ecosystem.png" alt="drawing" style="width:100%;"/> 

The FedML Ecosystem facilitates federated learning research and productization in diverse application domains. With the foundational support from FedML Core Framework, it supports FedNLP (Natural Language Processing), FedCV (Computer Vision), FedGraphNN (Graph Neural Networks), and FedIoT (Internet of Things).
Please read this guidance for details: [https://doc.fedml.ai/starter/ecosystem.html](https://doc.fedml.ai/starter/ecosystem.html)

### Publication
FedML’s core technology is backed by years of cutting-edge research represented in <b>50+ publications</b> in ML/FL Algorithms, Security/Privacy, Systems, and Applications.

1. Vision Paper for High Scientific Impacts
2. System for Large-scale Distributed/Federated Training
3. Training Algorithms for FL
4. Security/privacy for FL
5. AI Applications
A Full-stack of Scientific Publications in ML Algorithms, Security/Privacy, Systems, Applications, and Visionary Impacts

Please check out [the full publication list](./../resources/papers.md) for details.

### Invited Talks (Videos)

- [Trustworthy and Scalable Federated Learning](https://www.youtube.com/watch?v=U3BiuWjhdaU). Federated Learning One World Seminar (FLOW). By Salman Avestimehr

- [Distributed ML for Federated Learning feat](https://www.youtube.com/watch?v=AY7pCYTC8pQ). Chaoyang He. Stanford MLSys Seminar. By Chaoyang He

- [Contributed Talk for FedML Library](https://www.youtube.com/watch?v=93SETZGZMyI). Best Paper Award at NeurIPS 2020 Federated Learning Workshop. By Chaoyang He

### Join the Community

- Join our Slack:

<img src="./../_static/image/slack_logo.png" alt="drawing" style="width:200px;"/> 
<br />

[https://join.slack.com/t/fedml/shared_invite/zt-havwx1ee-a1xfOUrATNfc9DFqU~r34w](https://join.slack.com/t/fedml/shared_invite/zt-havwx1ee-a1xfOUrATNfc9DFqU~r34w)

- Join our WeChat Group:

Our WeChat group has 200+ members. Please add the following account and ask for an invitation to join. 

<img src="./../_static/image/wechat.jpeg" alt="drawing" style="width:200px;"/>

### FAQ

We organize the frequently asked questions at [https://github.com/FedML-AI/FedML/discussions](https://github.com/FedML-AI/FedML/discussions). 
Please feel free to ask questions there. We are happy to discuss supporting your special requests.

### Join Our Team
FedML is hiring researchers, engineers, product managers, and related interns.
If you are interested, please apply at [https://fedml.ai/careers](https://fedml.ai/careers).
