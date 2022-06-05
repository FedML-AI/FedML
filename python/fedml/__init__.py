import logging
import os
import random

import fedml
import numpy as np
import torch
import wandb

from .constants import (
    FEDML_TRAINING_PLATFORM_SIMULATION,
    FEDML_SIMULATION_TYPE_SP,
    FEDML_SIMULATION_TYPE_MPI,
    FEDML_SIMULATION_TYPE_NCCL,
    FEDML_TRAINING_PLATFORM_CROSS_SILO,
    FEDML_TRAINING_PLATFORM_CROSS_DEVICE,
)

from .cross_silo import Client as ClientCrossSilo
from .cross_silo import Server as ServerCrossSilo
from .cross_silo.hierarchical import Client as HierarchicalClientCrossSilo
from .cross_silo.hierarchical import Server as HierarchicalServerCrossSilo
from .core.mlops import MLOpsRuntimeLog
from .simulation.simulator import SimulatorMPI, SimulatorSingleProcess, SimulatorNCCL

_global_training_type = None
_global_comm_backend = None

__version__ = "0.7.77"


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
        args.training_type == FEDML_TRAINING_PLATFORM_SIMULATION
        and hasattr(args, "backend")
        and args.backend == "MPI"
    ):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        process_id = comm.Get_rank()
        worker_num = comm.Get_size()
        args.comm = comm
        args.process_id = process_id
        args.worker_num = worker_num
    elif (
        args.training_type == FEDML_TRAINING_PLATFORM_SIMULATION
        and hasattr(args, "backend")
        and args.backend == "single_process"
    ):
        pass
    elif args.training_type == "cross_silo":
        if not hasattr(args, "scenario"):
            args.scenario = "horizontal"
        if args.scenario == "horizontal":

            args.process_id = args.rank

        elif args.scenario == "hierarchical":
            args.worker_num = args.client_num_per_round
            if not hasattr(args, 'enable_cuda_rpc'):
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
                if not hasattr(args, 'pg_master_port'):
                    args.pg_master_port = 29200
                if not hasattr(args, 'pg_master_address'):
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
                


    elif args.training_type == "cross_device":
        args.rank = 0  # only server runs on Python package
    else:
        raise Exception("no such setting")
    return args


def run_simulation(backend=FEDML_SIMULATION_TYPE_SP):
    """FedML Parrot"""
    global _global_training_type
    _global_training_type = FEDML_TRAINING_PLATFORM_SIMULATION
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
    if backend == FEDML_SIMULATION_TYPE_SP:
        simulator = SimulatorSingleProcess(args, device, dataset, model)
    elif backend == FEDML_SIMULATION_TYPE_MPI:
        simulator = SimulatorMPI(args, device, dataset, model)
        logging.info("backend = {}".format(backend))
    elif backend == FEDML_SIMULATION_TYPE_NCCL:
        simulator = SimulatorNCCL(args, device, dataset, model)
        logging.info("backend = {}".format(backend))
    else:
        raise Exception("no such backend!")
    simulator.run()


def run_cross_silo_server():
    """FedML Octopus"""
    global _global_training_type
    _global_training_type = FEDML_TRAINING_PLATFORM_CROSS_SILO

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
    _global_training_type = FEDML_TRAINING_PLATFORM_CROSS_SILO

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
    global _global_training_type
    _global_training_type = FEDML_TRAINING_PLATFORM_CROSS_SILO

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
    global _global_training_type
    _global_training_type = FEDML_TRAINING_PLATFORM_CROSS_SILO
    
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
    from .cross_device import ServerMNN

    """FedML BeeHive"""
    global _global_training_type
    _global_training_type = FEDML_TRAINING_PLATFORM_CROSS_DEVICE

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


from .arguments import (
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
