# Getting Started

We require `peft>0.2.0` which is not officially released yet. Install from `peft` repo with

```shell
pip install git+https://github.com/huggingface/peft.git#egg=peft
```

# Centralized Training

```shell
bash scripts/train_ddp.sh

# train with DeepSpeed
bash scripts/train_deepspeed.sh
```

# Cross-silo Federated Learning

```shell
bash scripts/run_fedml_client.sh 1 fedllm
bash scripts/run_fedml_client.sh 2 fedllm
bash scripts/run_fedml_server.sh fedllm
```
