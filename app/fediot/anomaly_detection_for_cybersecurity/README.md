# FedIoT: Federated Learning for Internet of Things
This is the offical implementation of the paper: \
**Federated Learning for Internet of Things: A Federated Learning Framework for On-device Anomaly Data Detection** \
Tuo Zhang*, Chaoyang He*, Tianhao Ma, Lei Gao, Mark Ma, Salman Avestimehr \
(* means co-1st authors) \
accepted to ACM Embedded Networked Sensor Systems SenSys 2021 (AIChallengeIoT) \
[[Proceeding](https://dl.acm.org/doi/pdf/10.1145/3485730.3493444)] [[Arxiv](https://arxiv.org/abs/2106.07976)] \
TLDR: IoT x Federated Learning, from FedML.ai
## Introduction

Due to the heterogeneity, diversity, and personalization of IoT networks, Federated Learning (FL) has a promising future in the IoT cybersecurity field. As a result, we present the FedIoT, an open research platform and benchmark to facilitate FL research in the IoT field. In particular, we propose an autoencoder based trainer to IoT traffic data for anomaly detection. In addition, with the application of federated learning approach for aggregating, we propose an efficient and practical model for the anomaly detection in various types of devices, while preserving the data privacy for each device. What is more, our platform supports three diverse computing paradigms: 1) on-device training for IoT edge devices, 2) distributed computing, and 3) single-machine simulation to meet algorithmic and system-level research requirements under different system deployment scenarios. We hope FedIoT could provide an efficient and reproducible means for developing the implementation of FL in the IoT field. 

Check our slides [here](https://docs.google.com/presentation/d/1aW0GlOhKOl35jMl1KBDjKafJcYjWB-T9fiUsbdBySd4/edit?usp=sharing).
Learn more about Federated Learning for Internet of Things, please check our survey **Federated Learning for Internet of Things: Applications, Challenges, and Opportunities** at [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/9773116) and [Arxiv](https://arxiv.org/abs/2111.07494).


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
Install fedml library on serve side:
```
pip install fedml
```
Install fedml library on [raspberry](https://doc.fedml.ai/starter/install/rpi.html) and [jetson](https://doc.fedml.ai/starter/install/jetson.html) platforms.


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
```
@article{Zhang2021FederatedLF,
  title={Federated Learning for Internet of Things},
  author={Tuo Zhang and Chaoyang He and Tianhao Ma and Lei Gao and Mark Ma and Salman Avestimehr},
  journal={Proceedings of the 19th ACM Conference on Embedded Networked Sensor Systems},
  year={2021}
}
```

## Contact

Please find contact information at the [homepage](https://github.com/FedML-AI/FedML#join-the-community).
