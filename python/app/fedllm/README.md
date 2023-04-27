# FedLLM: Foundational Ecosystem Design for LLM

[FedLLM](https://blog.fedml.ai/releasing-fedllm-train-your-own-large-language-models-on-proprietary-data-using-the-fedml-platform/)
is an MLOps-supported training pipeline to help users build their own large language model on proprietary/private
data.
This repo aims to illustrate how to use FedML for large language model training and fine-tuning.

The repo contains:

- A minimalist PyTorch implementation for conventional/centralized LLM training, fine-tuning, and evaluation.
    - The training and evaluation logic
      uses [transformers.Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
    - LoRA integration from [peft](https://github.com/huggingface/peft)
    - Supports [DeepSpeed](https://www.deepspeed.ai/).
- Cross-silo Federated training/fine-tuning implementation with [FedML](https://github.com/FedML-AI/FedML).

## Getting Started

Install dependencies with the following command:

```shell
pip install -r requirements.txt
```

See [Dependencies](#dependencies) for more information on the dependency versions.

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

**_Notice_**: when using PyTorch DDP with LoRA and gradient checkpointing, you need to turn off `find_unused_parameters`
by passing `--ddp_find_unused_parameters "False"` in the command line.

### Cross-silo Federated Learning

To train/fine-tune in federated setting, you need to provide a FedML config file. An example can be found
in [fedml_config.yaml](fedml_config/fedml_config.yaml).
You can have different config file for each client or server.

Example scripts:

```shell
# run server
bash scripts/run_fedml_server.sh "$RUN_ID"

# run client(s)
for client_rank in "${client_ranks}"; do
  bash scripts/run_fedml_client.sh "$client_rank" "$RUN_ID"
done
```

See FedML's [Getting Started](https://doc.fedml.ai/starter/getting_started.html) for detail.

### Dependencies

Notice that we require `peft>=0.3.0` which is not officially released yet.
If you do not want to use our `requirements.txt`,
you can install `peft` from its repo with the following command:

```shell
pip install git+https://github.com/huggingface/peft.git#egg=peft
```

We have tested our implement with the following setup:

- Ubuntu `20.04.5 LTS`
- CUDA `11.8`, `11.7` and `11.6`
- Python `3.8.13`
    - `torch==0.2.0`
    - `torchvision==0.15.1`
    - `fedml==0.8.3`
    - `transformers==4.28.1`
    - `peft @ git+https://github.com/huggingface/peft.git@3890665e6082bdc354d4c3a26f3bf42ecedaaa81`
    - `datasets==2.11.0`
    - `deepspeed==0.9.1`
    - `numpy==1.24.3`
    - `tensorboard==2.12.2`
    - `mpi4py==3.1.4`
