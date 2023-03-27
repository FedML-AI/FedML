import logging
import os
import fedml
from .. import load_arguments, run_simulation, FEDML_TRAINING_PLATFORM_SIMULATION, FEDML_TRAINING_PLATFORM_CROSS_SILO, \
    collect_env, mlops


def init(args=None):
    print(f"args={args}")
    if args is None:
        args = load_arguments(training_type=None, comm_backend=None)

    """Initialize FedML Engine."""
    collect_env(args)
    fedml._global_training_type = args.training_type
    fedml._global_comm_backend = args.backend

    if args.training_type == FEDML_TRAINING_PLATFORM_SIMULATION and hasattr(args, "backend") and args.backend == "sp":
        args = init_simulation_sp(args)
    elif args.training_type == FEDML_TRAINING_PLATFORM_CROSS_SILO:
        args = init_cross_silo(args)
    else:
        raise Exception("no such setting: training_type = {}, backend = {}".format(args.training_type, args.backend))

    fedml.update_client_id_list(args)
    mlops.init(args)
    logging.info("args.rank = {}, args.worker_num = {}".format(args.rank, args.worker_num))
    fedml.update_client_specific_args(args)
    fedml.print_args(args)

    return args

def manage_mpi_args(args):
    if hasattr(args, "backend") and args.backend == "MPI":
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        process_id = comm.Get_rank()
        world_size = comm.Get_size()
        args.comm = comm
        args.rank = process_id
        if process_id == 0:
            args.role = "server"
        else:
            args.role = "client"
        assert args.worker_num + 1 == world_size, f"Invalid number of mpi processes. Expected {args.worker_num + 1}"
    else:
        args.comm = None

def init_cross_silo(args):
    manage_mpi_args(args)

    # Set intra-silo arguments
    if args.rank == 0:
        args.n_node_in_silo = 1
        args.n_proc_in_silo = 1
        args.rank_in_node = 0
        args.proc_rank_in_silo = 0
    else:
        # Modify arguments to match info set in env by torchrun
        # Silo Topology

        args.n_proc_in_silo = int(os.environ.get("WORLD_SIZE", 1))

        # Rank in node
        args.rank_in_node = int(os.environ.get("LOCAL_RANK", 0))
        args.process_id = args.rank_in_node

        # Rank in silo (process group)
        args.proc_rank_in_silo = int(os.environ.get("RANK", 0))

        # Process group master endpoint
        args.pg_master_address = os.environ.get("MASTER_ADDR", "127.0.0.1")
        args.pg_master_port = os.environ.get("MASTER_PORT", 29300)

        if not hasattr(args, "n_node_in_silo"):
            args.n_node_in_silo = 1
        if not (hasattr(args, "n_proc_per_node") and args.n_proc_per_node):
            args.n_proc_per_node = 1

    return args


def init_simulation_sp(args):
    return args


from .runner import FARunner

__all__ = [
    "FARunner",
    "run_simulation",
    "init"
]
