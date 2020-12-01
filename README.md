# FedML: A Research Library and Benchmark for Federated Machine Learning

<p align="center">
:page_facing_up: <a href="https://arxiv.org/abs/2007.13518">https://arxiv.org/abs/2007.13518</a>
</p>

<p align="center">
  <a href="https://github.com/FedML-AI/FedML/projects/1"><img alt="Roadmap" src="https://img.shields.io/badge/roadmap-FedML-informational.svg?style=flat-square"></a>
  <a href="#"><img alt="Python3" src="https://img.shields.io/badge/Python-3-brightgreen.svg?style=flat-square"></a>
  <a href="#"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%3E1.0-orange"></a>
  <a href="https://travis-ci.org/FedML-AI/FedML"><img alt="Travis" src="https://img.shields.io/travis/FedML-AI/FedML.svg?style=flat-square"></a>
  <a href="https://opencollective.com/fedml/donate"><img alt="Contributors" src="https://opencollective.com/fedml/tiers/badge.svg?style=flat-square"></a>
</p>

<p align="center">
   <a href="https://opencollective.com/fedml#support" target="_blank"><img src="https://opencollective.com/fedml/tiers/sponsors.svg?avatarHeight=36"></a>
</p>

## News
<b>2020-11-27 (System)</b>: FedML architecture has evolved into an ecosystem including multiple GitHub repositories. With FedML at its core, we can support more advanced FL applications and platforms. <br>
FedML: https://github.com/FedML-AI/FedML

FedNLP: https://github.com/FedML-AI/FedNLP

FedML-IoT: https://github.com/FedML-AI/FedML-IoT

FedML-Server: https://github.com/FedML-AI/FedML-Server

FedML-Mobile: https://github.com/FedML-AI/FedML-Mobile

<b>2020-11-24 (Publication)</b>: We are thrilled to share that the short version of our FedML white paper has been accepted to NeurIPS 2020 workshop. Thanks for reviewers from NeurIPS, supporting us to do a presentation there. <br>
<img src=https://github.com/FedML-AI/FedML/blob/master/docs/image/Neurips-logo.jpg width="35%">


<b>2020-11-05 (System)</b>: Do you want to run federated learning on <b>IoT devices</b>? FedML architecture design can smoothly transplant the distributed computing code to the IoT platform. FedML can support edge training on two IoT devices: <b>Raspberry Pi 4</b> and <b>NVIDIA Jetson Nano</b>!!! Please check it out here: https://github.com/FedML-AI/FedML/blob/master/fedml_iot/README.md <br>
<img src=https://github.com/FedML-AI/FedML/blob/master/docs/image/raspberry_pi.png width="35%">
<img src=https://github.com/FedML-AI/FedML/blob/master/docs/image/nvidia-jetson-nano.png width="35%">

<b>2020-10-28 (Algorithms) </b>: We released more advanced federated optimization algorithms, more than just FedAvg! http://doc.fedml.ai/#/algorithm-reference-implementation

<b>2020-10-26 (Publication) </b>: V2 of our white paper is released. Please check out here: https://arxiv.org/pdf/2007.13518.pdf

<b>2020-10-07 (Model and Dataset) </b>: Datasets + Models ALL IN ONE!!! FedML supports comprehensive research-oriented FL datasets and models:

- cross-device CV: Federated EMNIST + CNN (2 conv layers)
- cross-device CV: CIFAR100 + ResNet18 (Group Normalization)
- cross-device NLP: shakespeare + RNN (bi-LSTM)
- cross-device NLP: stackoverflow (NWP) + RNN (bi-LSTM)
- cross-silo CV: CIFAR10, CIFAR100, CINIC10 + ResNet
- cross-silo CV: CIFAR10, CIFAR100, CINIC10 + MobileNet
- linear: MNIST + Logistic Regression

Please check `create_model(args, model_name, output_dim)` and `load_data(args, dataset_name)` at `fedml_experiments/distributed/fedavg/main_fedavg.py` for details.

We will support more advanced models and datasets, please stay tuned!

---

<b>2020-09-30 (Publication)</b>: We maintained a comprehensive publication list of Federated Learning here: https://github.com/chaoyanghe/Awesome-Federated-Learning

---

<b>2020-09-28 (Publication)</b>: Authors of FedML (https://fedml.ai) have 7 papers that got accepted to NeurIPS 2020. Big congratulations!!!
Here is the publication list: https://github.com/FedML-AI/FedML/blob/master/publications.md. Highlighted ones are related to large-scale distributed learning and federated learning.


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

<img src="./docs/image/fedml.jpg" width="620">


The functionality of each package is as follows:

**fedml_core**: The FedML low level API package. This package implements distributed computing by communication backend like MPI, and also support topology management. 
Other low-level APIs related to security and privacy are also supported.

**fedml_api**: The FedML high level API package. This package support different federated learning algorithm with only one line code.
All algorithms are built based on the "fedml_core" package.
Users can change this package to add more advanced algorithms.


**fedml_experiments**: This package is used to test algorithms in "fedml" package by calling high level APIs.

**fedml_mobile**: This package is used to support on-device training using Android/iOS smartphones. 

**fedml_IoT**: This package is used to support on-device training using IoT devices. 

**applications**: This package is a collection of applications based on FedML.

**benchmark**: This package is used to run benchmark experiments.



## Join our Community
Please join our community. We will post updated features and answer questions on Slack.

[Join fedml.slack.com](https://join.slack.com/t/fedml/shared_invite/zt-havwx1ee-a1xfOUrATNfc9DFqU~r34w)
(this is a link that never expires)

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
