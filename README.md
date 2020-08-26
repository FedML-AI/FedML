<p align="center">

  <a href="https://github.com/FedML-AI/FedML/projects/1"><img alt="Roadmap" src="https://img.shields.io/badge/roadmap-FedML-informational.svg?style=flat-square"></a>
  <a href="#"><img alt="Python3" src="https://img.shields.io/badge/Python-3-brightgreen.svg?style=flat-square"></a>
  <a href="#"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%3E1.0-orange"></a>
  <a href="https://travis-ci.org/FedML-AI/FedML"><img alt="Travis" src="https://img.shields.io/travis/FedML-AI/FedML.svg?style=flat-square"></a>

</p>

Notice: *FedML is evolving. We will update more features in next 1-2 months. Please email us if there is misinformation.*

# FedML: A Research Library and Benchmark for Federated Machine Learning
[https://arxiv.org/abs/2007.13518](https://arxiv.org/abs/2007.13518)

## What is Federated Learning?
Please read this long vision paper [Advances and Open Problems in Federated Learning](https://arxiv.org/abs/1912.04977).

This publication list is also helpful: https://github.com/chaoyanghe/Awesome-Federated-Learning

## Introduction
Federated learning is a rapidly growing research field in the machine learning domain. 
Although considerable research efforts have been made, existing libraries cannot adequately support diverse algorithmic development (e.g., diverse topology and flexible message exchange), 
and inconsistent dataset and model usage in experiments make fair comparisons difficult.
In this work, we introduce FedML, an open research library and benchmark that facilitates the development of new federated learning algorithms and fair performance comparisons. 
FedML supports three computing paradigms (distributed training, mobile on-device training, and standalone simulation) for users to conduct experiments in different system environments. 
FedML also promotes diverse algorithmic research with flexible and generic API design and reference baseline implementations. A curated and comprehensive benchmark dataset for the non-I.I.D setting aims at making a fair comparison.
We believe FedML can provide an efficient and reproducible means of developing and evaluating algorithms for the federated learning research community. We maintain the source code, documents, and user community at https://FedML.ai.

For more details, please read our full paper: [https://arxiv.org/abs/2007.13518](https://arxiv.org/abs/2007.13518)

## Usage
1. Research on FL algorithm or system
2. Teaching in a ML course
3. System prototype for industrial production.
4. Self-study FL: understanding code level details of FL algorithms.

## Architecture

![architecture](./docs/image/architecture_for_website.png)


The functionality of each package is as follows:

**fedml_core**: The FedML low level API package. This package implements distributed computing by communication backend like MPI, and also support topology management. 
Other low-level APIs related to security and privacy are also supported.

**fedml**: The FedML high level API package. This package support different federated learning algorithm with only one line code.
All algorithms are built based on the "fedml_core" package.
Users can change this package to add more advanced algorithms.

**fedml_mobile**: This package is used to support on-device training using Android/iOS smartphones. 

**fedml_experiments**: This package is used to test algorithms in "fedml" package by calling high level APIs.

**benchmark**: This package is used to run benchmark experiments.

**applications**: This package is a collection of applications based on FedML.


## Join our Community
Please join our community. We will post updated features and answer questions on Slack.

[Join fedml.slack.com](https://join.slack.com/t/fedml/shared_invite/zt-gbpi8y2o-QMU0vhVHjm9Y9gqQu~eygg)


## Contributing
We sincerely welcome contributors and believe in the power of the open source. We welcome expertise from two tracks, either research or engineering.

1. If you are a researcher who needs APIs that our library does not support yet, please send us your valuable suggestions.

2. If you are a researcher who has published FL-related algorithm or system-level optimization, we welcome you to submit your source code to FedML, which will then be maintained by our engineers and researchers.

3. If you are an engineer or student who is searching for interesting open source projects to broaden your career, FedML is perfect for you. Currently, we are developing the following urgent features.

i) transplanting more advanced FL algorithms to FedML. We will show you some important research publications once you are involved. 
For this role, we prefer engineers or students who have a basic understanding of machine learning.

ii) FedML-Mobiel service architecture: Flask + PyTorch + RabbitMQ

iii) upgrading our Android and iOS platform.

iv) building or applying more models in computer vision and NLP domains to FedML.

v) collecting realistic federated datasets by crowdsourcing.

Please email us for further information. 

## Citation
Please cite FedML in your publications if it helps your research:
```
@article{chaoyanghe2020fedml,
  Author = {He, Chaoyang and Li, Songze and So, Jinhyun and Zhang, Mi and Wang, Hongyi and Wang, Xiaoyang and Vepakomma, Praneeth and Singh, Abhishek and Qiu, Hang and Shen, Li and Zhao, Peilin and Kang, Yan and Liu, Yang and Raskar, Ramesh and Yang, Qiang and Annavaram, Murali and Avestimehr, Salman},
  Journal = {arXiv preprint arXiv:2007.13518},
  Title = {FedML: A Research Library and Benchmark for Federated Machine Learning},
  Year = {2020}
}
```

## Contacts
The corresponding author is:
 
Chaoyang He\
chaoyang.he@usc.edu\
http://chaoyanghe.com
