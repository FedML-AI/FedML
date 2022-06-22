import copy
import logging
import random

import numpy as np
import torch
import wandb

from .my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from .my_model_trainer_nwp import MyModelTrainer as MyModelTrainerNWP
from .my_model_trainer_tag_prediction import MyModelTrainer as MyModelTrainerTAG
import logging


from ..base_framework.params import Params
from ..base_framework.params import ServerToClientParams
from ..base_framework.params import LocalAggregatorToServerParams
from ..base_framework.params import ClientToLocalAggregatorParams

from ..base_framework.common import fedml_nccl_broadcast
from ..base_framework.common import fedml_nccl_reduce
from ..base_framework.common import fedml_nccl_barrier
from ..base_framework.common import (get_server_rank, get_rank)

from ..base_framework.common import ReduceOp

from ..base_framework.common import (
    move_to_cpu, move_to_gpu,
    clear_optim_buffer, optimizer_to
)

from ..base_framework.Server import BaseServer



class FedAvgServer(BaseServer):
    # def __init__(self, args, trainer, device, dataset, comm=None, rank=0, size=0, backend="NCCL"):
    #     super().__init__(args, args, trainer, device, dataset, comm, rank, size, backend)
    def __init__(self, args, rank, worker_number, comm,
            device, dataset, model, trainer):
        super().__init__(args, rank, worker_number, comm,
            device, dataset, model, trainer)












