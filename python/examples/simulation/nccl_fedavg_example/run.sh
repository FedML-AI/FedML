#!/usr/bin/env bash


~/anaconda3/envs/py36/bin/python -m torch.distributed.launch \
    --nproc_per_node=10 --nnodes=1 --node_rank=0 \
    --master_addr localhost --master_port 11111 \
    torch_fedavg_mnist_lr_custum_data_and_model_example.py --cf config/fedml_config.yaml