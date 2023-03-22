# FedML - The community building and connecting AI anywhere at any scale

https://doc.fedml.ai

## Mission
FedML builds simple and versatile APIs for machine learning running anywhere at any scale.
In other words, FedML supports both federated learning for data silos and distributed training for acceleration with MLOps and Open Source support, covering industrial grade use cases and cutting-edge academia research.

- Distributed Training: Accelerate Model Training with Lightweight Cheetah
- Simulator: (1) simulate FL using a single process (2) MPI-based FL Simulator (3) NCCL-based FL Simulator (fastest)
- Cross-silo Federated Learning for cross-organization/account training, including Python-based edge SDK
- Cross-device Federated Learning for Smartphones and IoTs, including edge SDK for Android/iOS and embedded Linux.
- Model Serving: we focus on providing a better user experience for edge AI.
- MLOps: FedML's machine learning operation pipeline for AI running anywhere at any scale.

## Source Code Structure


The functionality of each package is as follows:

**core**: The FedML low-level API package. This package implements distributed computing by communication backend like MPI, NCCL, MQTT, gRPC, PyTorch RPC, and also supports topology management. 
Other low-level APIs related to security and privacy are also supported. All algorithms and Scenarios are built based on the "core" package.

**data**: FedML will provide some default datasets for users to get started. Customization templates are also provided.

**model**: FedML model zoo.

**device**: FedML computing resource management.

**simulation**: FedML parrot can support: (1) simulate FL using a single process (2) MPI-based FL Simulator (3) NCCL-based FL Simulator (fastest)

**cross-silo**: Cross-silo Federated Learning for cross-organization/account training

**cross-device**: Cross-device Federated Learning for Smartphones and IoTs

**distributed**: Distributed Training: Accelerate Model Training with Lightweight Cheetah

**serve**: Model serving, tailored for edge inference

**mlops**: APIs related to machine learning operation platform (open.fedml.ai)

**centralized**: Some centralized trainer code examples for benchmarking purposes.

**utils**: Common utilities shared by other modules.

## About FedML, Inc.
https://FedML.ai
