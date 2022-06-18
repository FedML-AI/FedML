import copy
import logging
import random

import numpy as np
import torch
import wandb

import logging

from .my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from .my_model_trainer_nwp import MyModelTrainer as MyModelTrainerNWP
from .my_model_trainer_tag_prediction import MyModelTrainer as MyModelTrainerTAG

from .LocalAggregator import FedAvgLocalAggregator
from .Server import FedAvgServer


def FedML_FedAvg_NCCL(args, process_id, worker_number, comm,
            device, dataset, model, model_trainer=None):
    if model_trainer is None:
        if args.dataset == "stackoverflow_lr":
            model_trainer = MyModelTrainerTAG(model)
        elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp"]:
            model_trainer = MyModelTrainerNWP(model)
        else:  # default model trainer is for classification problem
            model_trainer = MyModelTrainerCLS(model)
    if process_id == 0:
        return FedAvgServer(args, process_id, worker_number, comm,
            device, dataset, model, model_trainer)
    else:
        return FedAvgLocalAggregator(args, process_id, worker_number, comm,
            device, dataset, model, model_trainer)


# def FedML_FedAvg_NCCL(args, process_id, worker_number, comm):
#     if process_id == 0:
#         return init_server(args, comm, process_id, worker_number)
#     else:
#         return init_local_aggregator(args, comm, process_id, worker_number)


# def init_server(args, comm, process_id, size):
#     # aggregator
#     client_num = size - 1
#     server = FedAvgServer(client_num, args)
#     return server


# def init_local_aggregator(args, comm, process_id, size):
#     # trainer
#     client_ID = process_id - 1
#     local_aggregator = FedAvgLocalAggregator()
#     return local_aggregator











