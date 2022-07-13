import copy
import logging
import random

import numpy as np
import torch
import wandb

import logging


from .params import Params
from .params import ServerToClientParams
from .params import LocalAggregatorToServerParams
from .params import ClientToLocalAggregatorParams

from .LocalAggregator import BaseLocalAggregator
from .Server import BaseServer


from .common import fedml_nccl_broadcast
from .common import fedml_nccl_reduce
from .common import fedml_nccl_barrier
from .common import (get_server_rank, get_rank)

from .common import ReduceOp

from .common import Role, CommState
from .common import FedML_NCCL_Similulation_init

from .common import (
    move_to_cpu, move_to_gpu,
    clear_optim_buffer, optimizer_to
)





def FedML_Base_NCCL(args, process_id, worker_number, comm,
            device, dataset, model, model_trainer=None):

    if process_id == 0:
        return BaseServer(args, process_id, worker_number, comm,
            device, dataset, model, model_trainer)
    else:
        return BaseLocalAggregator(args, process_id, worker_number, comm,
            device, dataset, model, model_trainer)











