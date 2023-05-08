<div align="center">
 <img src="assets/fedml_logo_light_mode.png" width="400px">
</div>

# FedLLM: Build Your Own Large Language Models on Proprietary Data using the FedML Platform

[FedLLM](https://blog.fedml.ai/releasing-fedllm-build-your-own-large-language-models-on-proprietary-data-using-the-fedml-platform/)
is an MLOps-supported training pipeline to help users build their own large language model (LLM) on proprietary/private
data.
This repo aims to provide a minimalist example of efficient LLM training/fine-tuning
and to illustrate how to use FedML for federated LLM training and fine-tuning.

The repo contains:

- A minimalist PyTorch implementation for conventional/centralized LLM training, fine-tuning, and evaluation.
    - The training and evaluation logic
      follows [transformers.Trainer](https://huggingface.co/docs/transformers/main_classes/trainer).
    - LoRA integration from [peft](https://github.com/huggingface/peft).
    - Supports [DeepSpeed](https://www.deepspeed.ai/).
    - Dataset implementation with [datasets](https://huggingface.co/docs/datasets/index).
- Cross-silo Federated training/fine-tuning implementation with [FedML](https://github.com/FedML-AI/FedML).

## Getting Started

Install dependencies with the following command:

```shell
pip install -r requirements.txt
```

See [Dependencies](#dependencies) for more information on the dependency versions.

### Prepare Dataset

Run the following command to download [`databricks-dolly-15k`](https://github.com/databrickslabs/dolly/tree/master/data).

```shell
bash script/setup.sh
```

### Conventional/Centralized Training

The [`train.py`](train.py) contains a minimal example for conventional/centralized LLM training and fine-tuning
on [`databricks-dolly-15k`](https://github.com/databrickslabs/dolly/tree/master/data) dataset.

Example scripts:

```shell
# train on a single GPU
bash scripts/train.sh \
  ... # additional arguments

# train with PyTorch DDP
bash scripts/train_ddp.sh \
  ... # additional arguments

# train with DeepSpeed
bash scripts/train_deepspeed.sh \
  ... # additional arguments
```

**_Tips_**: if you have an Amper or newer GPU (e.g., RTX 3000 or newer), you could turn on **bf16** to have more efficient training by passing
`--bf16 "True"` in the command line.

**_Notice_**: when using PyTorch DDP with LoRA and gradient checkpointing,
you need to turn off `find_unused_parameters`
by passing `--ddp_find_unused_parameters "False"` in the command line.

### Cross-silo Federated Learning

To train/fine-tune in federated setting, you need to provide a FedML config file.
An example can be found in [fedml_config.yaml](fedml_config/fedml_config.yaml).
You can have different config file for each client or server.
To launch an experiment, a `RUN_ID` should be provided. For each experiment, the same `RUN_ID` should be used across all
the client(s) and server.

**_Notice_**: since we use `RUN_ID` to uniquely identify experiments,
we recommend that you carefully choose the `RUN_ID`.
You may also generate a UUID for your `RUN_ID` with built-in Python module `uuid`;
e.g. use `RUN_ID="$(python3 -c "import uuid; print(uuid.uuid4().hex)")"` in your shell script.

Example scripts:

```shell
# run server
bash scripts/run_fedml_server.sh "$RUN_ID"

# run client(s)
for client_rank in "${client_ranks}"; do
  bash scripts/run_fedml_client.sh "$client_rank" "$RUN_ID" &
done
```

See FedML's [Getting Started](https://doc.fedml.ai/starter/getting_started.html) for detail.

### Dependencies

We have tested our implement with the following setup:

- Ubuntu `20.04.5 LTS`
- CUDA `11.8`, `11.7` and `11.6`
- Python `3.8.13`
    - `fedml>=0.8.4a7,<=0.8.4a17`
    - `torch==0.2.0`
    - `torchvision==0.15.1`
    - `transformers==4.28.1`
    - `peft==0.3.0`
    - `datasets==2.11.0`
    - `deepspeed==0.9.1`
    - `numpy==1.24.3`
    - `tensorboard==2.12.2`
    - `mpi4py==3.1.4`
