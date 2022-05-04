import logging
import random

import fedml
import numpy as np
import torch
import wandb
from fedml.mlops import MLOpsRuntimeLog
from mpi4py import MPI
import os
from .cross_device import ServerMNN
from .cross_silo import Client as ClientCrossSilo
from .cross_silo import Server as ServerCrossSilo
from .cross_silo.hierarchical import Client as HierarchicalClientCrossSilo
from .cross_silo.hierarchical import Server as HierarchicalServerCrossSilo
from .simulation.simulator import SimulatorMPI, SimulatorSingleProcess, SimulatorNCCL

_global_training_type = None
_global_comm_backend = None


def init(args=None):
    """Initialize FedML Engine."""
    global _global_training_type
    global _global_comm_backend

    if args is None:
        args = load_arguments(_global_training_type, _global_comm_backend)

    MLOpsRuntimeLog.get_instance(args).init_logs()

    logging.info("args = {}".format(vars(args)))

    seed = args.random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    if args.enable_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=args,
        )

    if (
        args.training_type == "simulation"
        and hasattr(args, "backend")
        and args.backend == "MPI"
    ):
        comm = MPI.COMM_WORLD
        process_id = comm.Get_rank()
        worker_num = comm.Get_size()
        args.comm = comm
        args.process_id = process_id
        args.worker_num = worker_num
    elif (
        args.training_type == "simulation"
        and hasattr(args, "backend")
        and args.backend == "single_process"
    ):
        pass
    elif args.training_type == "cross_silo":
        if not hasattr(args, 'enable_cuda_rpc'):
            args.enable_cuda_rpc = False
        # Set inra-silo argiments
        if args.rank == 0:
            args.rank_in_node = 0
            args.process_id = args.rank_in_node
            args.n_proc_in_silo = 1
            args.proc_rank_in_silo = 0
        else:
            args.rank_in_node = int(os.environ["LOCAL_RANK"])
            args.process_id = args.rank_in_node
            args.n_proc_in_silo = args.n_node_in_silo * args.n_proc_per_node
            args.proc_rank_in_silo = args.node_rank_in_silo * args.n_proc_per_node + args.rank_in_node
            args.pg_master_port += args.rank
    elif args.training_type == "cross_device":
        args.rank = 0  # only server runs on Python package
    else:
        raise Exception("no such setting")
    return args


def run_simulation(backend="single_process"):
    """FedML Parrot"""
    global _global_training_type
    _global_training_type = "simulation"
    global _global_comm_backend
    _global_comm_backend = backend

    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model
    model = fedml.model.create(args, output_dim)

    # start training
    if backend == "single_process":
        simulator = SimulatorSingleProcess(args, device, dataset, model)
    elif backend == "MPI":
        simulator = SimulatorMPI(args, device, dataset, model)
        logging.info("backend = {}".format(backend))
    elif backend == "NCCL":
        simulator = SimulatorNCCL(args, device, dataset, model)
        logging.info("backend = {}".format(backend))
    else:
        raise Exception("no such backend!")
    simulator.run()


def run_cross_silo_server():
    """FedML Octopus"""
    global _global_training_type
    _global_training_type = "cross_silo"

    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model
    model = fedml.model.create(args, output_dim)

    # start training
    server = ServerCrossSilo(args, device, dataset, model)
    server.run()


def run_cross_silo_client():
    """FedML Octopus"""
    global _global_training_type
    _global_training_type = "cross_silo"

    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model
    model = fedml.model.create(args, output_dim)

    # start training
    client = ClientCrossSilo(args, device, dataset, model)
    client.run()


def run_hierarchical_cross_silo_server():
    """FedML Octopus"""
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model
    model = fedml.model.create(args, output_dim)

    # start training
    server = HierarchicalServerCrossSilo(args, device, dataset, model)
    server.run()


def run_hierarchical_cross_silo_client():
    """FedML Octopus"""
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load_cross_silo(args)

    # load model
    model = fedml.model.create(args, output_dim)

    # start training
    client = HierarchicalClientCrossSilo(args, device, dataset, model)
    client.run()



def run_mnn_server():
    """FedML BeeHive"""
    global _global_training_type
    _global_training_type = "cross_device"

    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model
    model = fedml.model.create(args, output_dim)

    # start training
    server = ServerMNN(args, device, dataset, model)
    server.run()


def run_distributed():
    pass


from fedml.arguments import (
    load_arguments,
)

from .core.alg_frame.client_trainer import ClientTrainer
from .core.alg_frame.server_aggregator import ServerAggregator

from fedml import device
from fedml import data
from fedml import model
from fedml import simulation
from fedml import cross_silo
from fedml import cross_device


__all__ = [
    "device",
    "data",
    "model",
    "simulation",
    "cross_silo",
    "cross_device",
    "ClientTrainer",
    "ServerAggregator",
]
