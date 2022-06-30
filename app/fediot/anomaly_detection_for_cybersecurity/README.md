# FedIoT: Federated Learning for Internet of Things
 
<!-- This is FedIoT, an application ecosystem for federated IoT based on FedML framework (https://github.com/FedML-AI/FedML). -->

This repository is the official implementation of Federated Learning for Internet of Things: A Federated Learning Framework for On-device Anomaly Data Detection.
Read our paper here: https://arxiv.org/abs/2106.07976
## Introduction

Due to the heterogeneity, diversity, and personalization of IoT networks, Federated Learning (FL) has a promising future in the IoT cybersecurity field. As a result, we present the FedIoT, an open research platform and benchmark to facilitate FL research in the IoT field. In particular, we propose an autoencoder based trainer to IoT traffic data for anomaly detection. In addition, with the application of federated learning approach for aggregating, we propose an efficient and practical model for the anomaly detection in various types of devices, while preserving the data privacy for each device. What is more, our platform supports three diverse computing paradigms: 1) on-device training for IoT edge devices, 2) distributed computing, and 3) single-machine simulation to meet algorithmic and system-level research requirements under different system deployment scenarios. We hope FedIoT could provide an efficient and reproducible means for developing the implementation of FL in the IoT field. 

Check our slides here: https://docs.google.com/presentation/d/1aW0GlOhKOl35jMl1KBDjKafJcYjWB-T9fiUsbdBySd4/edit?usp=sharing

## Prerequisite
Install command line tool `unar` on both server and client side: 

for MacOS
```
brew install unar
```
for Linux
```
sudo apt-get install unar
```

## Real-deployment Training Script

At the client side, the client ID (a.k.a rank) starts from 1.
Please also modify config/fedml_config.yaml, changing the `client_num_in_total`, `client_num_per_round`, `worker_num` 
as the number of clients you plan to run.

For this application, the number of clients is up to 9 since there are 9 types of IoT devices in N-BaIoT dataset.

At the server side, run the following script:
```
bash run_server.sh
```

For client 1, run the following script:
```
bash run_client.sh 1
```
For client 2, run the following script:
```
bash run_client.sh 2
```
Note: please run the server first.

## Centralized Simulation Training Script

We also support centralized simulation for FedIoT, which means one PC could simulate both server and clients as you need.
All training related parameters are inside config_simulation/fedml_config.yaml, please modify it per your need.
The worker_num under the device_args represents number of processes in MPI, as the number of parallel clients.

For this application, the number of clients is up to 9 since there are 9 types of IoT devices in N-BaIoT dataset.
Run the following script to begin the training:
```
sh run_simulation.sh 9
```
9 in the above script represents the number of parallel clients, which should be identical to the worker_num inside config_simulation/fedml_config.yaml.

## A Better User-experience with FedML MLOps (open.fedml.ai)
To reduce the difficulty and complexity of these CLI commands. We recommend you to use our MLOps (open.fedml.ai).
FedML MLOps provides:
- Install Client Agent and Login
- Inviting Collaborators and group management
- Project Management
- Experiment Tracking (visualizing training results)
- monitoring device status
- visualizing system performance (including profiling flow chart)
- distributed logging
- model serving

## Citation
Please cite our FedIoT and FedML paper if it helps your research.
You can describe us in your paper like this: "We develop our experiments based on FedIoT [1] and FedML [2]".

@article{Zhang2021FederatedLF,
  title={Federated Learning for Internet of Things: A Federated Learning Framework for On-device Anomaly Data Detection},
  author={Tuo Zhang and Chaoyang He and Tian-Shya Ma and Mark Ma and S. Avestimehr},
  journal={ArXiv},
  year={2021},
  volume={abs/2106.07976}
}

## Contact

The corresponding author is:

Tuo Zhang
tuozhang@usc.edu

Chaoyang He
chaoyang.he@usc.edu
http://chaoyanghe.com

Lei Gao
leig@usc.edu
