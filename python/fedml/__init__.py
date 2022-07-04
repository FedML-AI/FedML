import logging
import multiprocessing
import os
import random

import numpy as np
import torch
import wandb

import fedml
from .constants import (
    FEDML_TRAINING_PLATFORM_SIMULATION,
    FEDML_SIMULATION_TYPE_SP,
    FEDML_SIMULATION_TYPE_MPI,
    FEDML_SIMULATION_TYPE_NCCL,
    FEDML_TRAINING_PLATFORM_CROSS_SILO,
    FEDML_TRAINING_PLATFORM_CROSS_DEVICE,
)
from .core.mlops import MLOpsRuntimeLog

_global_training_type = None
_global_comm_backend = None

__version__ = "0.7.111"


def init(args=None):
    """Initialize FedML Engine."""

    if args is None:
        args = load_arguments(fedml._global_training_type, fedml._global_comm_backend)

    fedml._global_training_type = args.training_type
    fedml._global_comm_backend = args.backend

    if (
        hasattr(args, "enable_tracking")
        and args.enable_tracking is True
        and args.training_type == FEDML_TRAINING_PLATFORM_SIMULATION
    ):
        mlops.init(args)
    else:
        MLOpsRuntimeLog.get_instance(args).init_logs()

    logging.info("args = {}".format(vars(args)))

    """
    # Windows/Linux/MacOS compatability issues on multi-processing
    # https://github.com/pytorch/pytorch/issues/3492
    """
    if multiprocessing.get_start_method() != "spawn":
        # force all platforms (Windows/Linux/MacOS) to use the same way (spawn) for multiprocessing
        multiprocessing.set_start_method("spawn", force=True)

    seed = args.random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    if args.enable_wandb:
        wandb.init(
            project=args.wandb_project, name=args.run_name, config=args,
        )

    if (
        args.training_type == FEDML_TRAINING_PLATFORM_SIMULATION
        and hasattr(args, "backend")
        and args.backend == "MPI"
    ):
        args = init_simulation_mpi(args)

    elif (
        args.training_type == FEDML_TRAINING_PLATFORM_SIMULATION
        and hasattr(args, "backend")
        and args.backend == "sp"
    ):
        args = init_simulation_sp(args)
    elif (
        args.training_type == FEDML_TRAINING_PLATFORM_SIMULATION
        and hasattr(args, "backend")
        and args.backend == FEDML_SIMULATION_TYPE_NCCL
    ):
        from .simulation.nccl.base_framework.common import FedML_NCCL_Similulation_init

        args = FedML_NCCL_Similulation_init(args)

    elif args.training_type == FEDML_TRAINING_PLATFORM_CROSS_SILO:
        if not hasattr(args, "scenario"):
            args.scenario = "horizontal"
        if args.scenario == "horizontal":

            args = init_cross_silo_horizontal(args)

        elif args.scenario == "hierarchical":
            args = init_cross_silo_hierarchical(args)

    elif args.training_type == FEDML_TRAINING_PLATFORM_CROSS_DEVICE:
        args = init_cross_device(args)
    else:
        raise Exception("no such setting")
    return args


def init_simulation_mpi(args):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_num = comm.Get_size()
    args.comm = comm
    args.process_id = process_id
    args.worker_num = worker_num
    return args


def init_simulation_sp(args):
    return args


def init_simulation_nccl(args):
    return


def init_cross_silo_horizontal(args):
    args.process_id = args.rank

    if hasattr(args, "backend") and args.backend == "MPI":
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        process_id = comm.Get_rank()
        worker_num = comm.Get_size()
        args.comm = comm
        args.process_id = process_id
        args.worker_num = worker_num - 1
        logging.info("comm = {}".format(comm))

    else:
        args.comm = None
    return args


def init_cross_silo_hierarchical(args):
    args.worker_num = args.client_num_per_round
    if not hasattr(args, "enable_cuda_rpc"):
        args.enable_cuda_rpc = False
    # Set intra-silo arguments
    if args.rank == 0:
        # Silo Topology
        if not hasattr(args, "n_proc_per_node"):
            args.n_proc_per_node = 1
        args.n_proc_in_silo = 1

        # Rank in node
        args.rank_in_node = 0
        args.process_id = args.rank_in_node

        # Rank in silo (process group)
        args.proc_rank_in_silo = 0

        # Prcoess group master endpoint
        if not hasattr(args, "pg_master_port"):
            args.pg_master_port = 29200
        if not hasattr(args, "pg_master_address"):
            args.pg_master_address = "127.0.0.1"
    else:
        # Modify arguments to match info set in env by torchrun
        # Silo Topology
        if not hasattr(args, "n_node_in_silo"):
            args.n_node_in_silo = 1
        if not hasattr(args, "n_proc_per_node"):
            args.n_proc_per_node = 1
        args.n_proc_in_silo = int(os.environ.get("WORLD_SIZE", 1))

        # Rank in node
        args.rank_in_node = int(os.environ.get("LOCAL_RANK", 1))
        args.process_id = args.rank_in_node

        # Rank in silo (process group)
        args.proc_rank_in_silo = int(os.environ.get("RANK", 0))

        # Prcoess group master endpoint
        args.pg_master_address = os.environ.get("MASTER_ADDR", "127.0.0.1")
        args.pg_master_port = os.environ.get("MASTER_PORT", 29300)

        # Launcher Rendezvous
        if not hasattr(args, "launcher_rdzv_port"):
            args.launcher_rdzv_port = 29400
    return args


def init_cross_device(args):
    args.rank = 0  # only server runs on Python package
    return args


def run_distributed():
    pass


from fedml import device
from fedml import data
from fedml import model
from fedml import mlops

from .arguments import load_arguments

from .core.alg_frame.client_trainer import ClientTrainer
from .core.alg_frame.server_aggregator import ServerAggregator

from .launch_simulation import run_simulation

from .launch_cross_silo_horizontal import run_cross_silo_server
from .launch_cross_silo_horizontal import run_cross_silo_client

from .launch_cross_silo_hi import run_hierarchical_cross_silo_server
from .launch_cross_silo_hi import run_hierarchical_cross_silo_client

from .launch_cross_device import run_mnn_server

__all__ = [
    "device",
    "data",
    "model",
    "mlops",
    "ClientTrainer",
    "ServerAggregator",
    "run_simulation",
    "run_cross_silo_server",
    "run_cross_silo_client",
    "run_hierarchical_cross_silo_server",
    "run_hierarchical_cross_silo_client",
    "run_mnn_server",
]
