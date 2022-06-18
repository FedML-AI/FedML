from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
)
import copy
import logging
import math
import os

import numpy as np
import sklearn
import torch
import wandb
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {"Total": total_num, "Trainable": trainable_num}


def freeze_model_parameters(model, freeze_layers):
    modules = list()
    logging.info("freeze layers: %s" % str(freeze_layers))
    for layer_idx in freeze_layers:
        if layer_idx == "e":
            modules.append(model.distilbert.embeddings)
        else:
            modules.append(model.distilbert.transformer.layer[int(layer_idx)])
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False
    logging.info(get_parameter_number(model))


def build_optimizer(model, iteration_in_total, args):
    warmup_steps = math.ceil(iteration_in_total * args.warmup_ratio)
    args.warmup_steps = warmup_steps if args.warmup_steps == 0 else args.warmup_steps
    logging.info("warmup steps = %d" % args.warmup_steps)
    # freeze exps only apply for distilbert
    # if args.model_type == "distilbert":
    #    freeze_model_parameters(model)
    if args.optimizer == "AdamW":
        optimizer = AdamW(
            model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon
        )
    else:
        optimizer = SGD(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=iteration_in_total,
    )
    return optimizer, scheduler
