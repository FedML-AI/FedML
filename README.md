# FedML: A Research Library and Benchmark for Federated Machine Learning
http://fedml.ai

## What is Federated Learning?
Please read this long vision paper "Advances and Open Problems in Federated Learning" (https://arxiv.org/abs/1912.04977).

This publication list is also helpful: https://github.com/chaoyanghe/Awesome-Federated-Learning

## Introduction
FedML is a research-oriented federated learning software library. Its primary purpose is to provide researchers and engineers with a flexible and generic experiment platform for developing innovative learning algorithms. It offers a worker-oriented programming interface, reference implementations for baselines, and a curated and comprehensive benchmark datasets for the non-I.I.D setting. With \texttt{FedML}, users can attach any behavior to workers in the arbitrary FL network (e.g., training, aggregation,  attack, and defense, etc.), customize any additional exchanging information, and control information flow among workers. Its reference implementations, benchmark, and datasets aim to promote fast reproducibility for baselines and fair comparison for newly developed algorithms. In this article, we describe the system design, the new programming interface, application examples, benchmark, dataset, and some experimental results. We accept user feedback, and will continuously update our library to support more advanced requirements.

For more details, please read our paper.

## Usage
1. Research on FL algorithm or system
2. Teaching in a ML course
3. System prototype for industrial production.
4. Self-study FL: understanding code level details of FL algorithms.

## Architecture

![architecture](./img/architecture_for_website.png)


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

## Installation
#### 1. Standalone Mode
Under construction

#### 2. Distributed Computing Mode
Please check the README.md in this directory: ./fedml_experiments/distributed/fedavg/README.md

#### 3. Mobile Device Training Mode
Under construction

## Join our Community
Join slack.fedml.ai

Join our WeChat Group



## Contributing
We sincerely welcome contributors. Please read this page to know how to start contributing code to FedML. 

## Citation
Please cite FedML in your publications if it helps your research:
```
@article{chaoyanghe2020fedml,
  Author = {Chaoyang He},
  Journal = {arXiv preprint arXiv:1802.05799},
  Title = {FedML: a Flexible and Generic Federated Learning Library and Benchmark},
  Year = {2020}
}
```

## Contacts
The corresponding author is:
 
Chaoyang He\
chaoyang.he@usc.edu\
http://chaoyanghe.com
