# Open Source Code Architecture
In general, FedML source code architecture follows the paper which won [Best Paper Award at NeurIPS 2020 (FL workshop)](https://chaoyanghe.com/wp-content/uploads/2021/02/NeurIPS-SpicyFL-2020-Baidu-best-paper-award-He-v2.pdf). Its original idea is presented at the live [video](https://www.youtube.com/watch?v=93SETZGZMyI) and 
[white paper](https://arxiv.org/abs/2007.13518) by FedML co-founder Dr. [Chaoyang He](https://chaoyanghe.com). 

![FedML Code Architecture](../_static/image/fedml.png)

### Code Architecture
After March 2022, FedML has been upgraded as an AI company which aims to provide machine learning capability anywhere at any scale. Now the python version of FedML [https://github.com/FedML-AI/FedML-refactor/tree/master/python](https://github.com/FedML-AI/FedML-refactor/tree/master/python) is reorganized as follows:

**core**: The FedML low-level API package. This package implements distributed computing by communication backend like MPI, NCCL, MQTT, gRPC, PyTorch RPC, and also supports topology management. 
Other low-level APIs related to security and privacy are also supported. All algorithms and Scenarios are built based on the "core" package.

**data**: FedML will provide some default datasets for users to get started. Customization templates are also provided.

**model**: FedML model zoo.

**device**: FedML computing resource management.

**simulation**: FedML parrot can support (1) simulate FL using a single process (2) MPI-based FL Simulator (3) NCCL-based FL Simulator (fastest)

**cross-silo**: Cross-silo Federated Learning for cross-organization/account training

**cross-device**: Cross-device Federated Learning for Smartphones and IoTs

**distributed**: Distributed Training: Accelerate Model Training with Lightweight Cheetah

**serve**: Model serving, tailored for edge inference

**mlops**: APIs related to machine learning operation platform (open.fedml.ai)

**centralized**: Some centralized trainer code examples for benchmarking purposes.

**utils**: Common utilities shared by other modules.

### Reference
```
@article{chaoyanghe2020fedml,
  Author = {He, Chaoyang and Li, Songze and So, Jinhyun and Zhang, Mi and Wang, Hongyi and Wang, Xiaoyang and Vepakomma, Praneeth and Singh, Abhishek and Qiu, Hang and Shen, Li and Zhao, Peilin and Kang, Yan and Liu, Yang and Raskar, Ramesh and Yang, Qiang and Annavaram, Murali and Avestimehr, Salman},
  Journal = {Advances in Neural Information Processing Systems, Best Paper Award at Federate Learning Workshop},
  Title = {FedML: A Research Library and Benchmark for Federated Machine Learning},
  Year = {2020}
}
```
### Video

[http://www.youtube.com/watch?v=93SETZGZMyI](http://www.youtube.com/watch?v=93SETZGZMyI)