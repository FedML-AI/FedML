import logging
import os
import fedml
from .. import load_arguments, run_simulation, FEDML_TRAINING_PLATFORM_SIMULATION, FEDML_TRAINING_PLATFORM_CROSS_SILO, \
    collect_env, mlops, FEDML_TRAINING_PLATFORM_CROSS_DEVICE


def init(args=None):
    print(f"args={args}")
    if args is None:
        args = load_arguments(training_type=None, comm_backend=None)

    """Initialize FedML Engine."""
    collect_env()
    fedml._global_training_type = args.training_type
    fedml._global_comm_backend = args.backend

    if args.training_type == FEDML_TRAINING_PLATFORM_SIMULATION and hasattr(args, "backend") and args.backend == "sp":
        args = init_simulation_sp(args)
    elif args.training_type == FEDML_TRAINING_PLATFORM_CROSS_SILO:
        args = init_cross_silo(args)
    else:
        raise Exception("no such setting: training_type = {}, backend = {}".format(args.training_type, args.backend))

    _update_client_id_list(args)
    mlops.init(args)
    logging.info("args.rank = {}, args.worker_num = {}".format(args.rank, args.worker_num))
    _update_client_specific_args(args)
    _print_args(args)

    return args


def _print_args(args):
    mqtt_config_path = None
    s3_config_path = None
    args_copy = args
    if hasattr(args_copy, "mqtt_config_path"):
        mqtt_config_path = args_copy.mqtt_config_path
        args_copy.mqtt_config_path = ""
    if hasattr(args_copy, "s3_config_path"):
        s3_config_path = args_copy.s3_config_path
        args_copy.s3_config_path = ""
    logging.info("==== args = {}".format(vars(args_copy)))
    if hasattr(args_copy, "mqtt_config_path"):
        args_copy.mqtt_config_path = mqtt_config_path
    if hasattr(args_copy, "s3_config_path"):
        args_copy.s3_config_path = s3_config_path


def _update_client_specific_args(args):
    """
        data_silo_config is used for reading specific configuration for each client
        Example: In fedml_config.yaml, we have the following configuration
        client_specific_args:
            data_silo_config:
                [
                    fedml_config/data_silo_1_config.yaml,
                    fedml_config/data_silo_2_config.yaml,
                    fedml_config/data_silo_3_config.yaml,
                    fedml_config/data_silo_4_config.yaml,
                ]
            data_silo_1_config.yaml contains some client client speicifc arguments.
    """
    if (
            hasattr(args, "data_silo_config")
    ):
        # reading the clients file
        logging.info("data_silo_config is defined in fedml_config.yaml")
        args.rank = int(args.rank)
        args.worker_num = len(args.data_silo_config)
        if args.rank > 0:
            extra_config_path = args.data_silo_config[args.rank - 1]
            extra_config = args.load_yaml_config(extra_config_path)
            args.set_attr_from_config(extra_config)
    else:
        logging.info("data_silo_config is not defined in fedml_config.yaml")


def _update_client_id_list(args):
    """
        generate args.client_id_list for CLI mode where args.client_id_list is set to None
        In MLOps mode, args.client_id_list will be set to real-time client id list selected by UI (not starting from 1)
    """
    if not hasattr(args, "using_mlops") or (hasattr(args, "using_mlops") and not args.using_mlops):
        if not hasattr(args,
                       "client_id_list") or args.client_id_list is None or args.client_id_list == "None" or args.client_id_list == "[]":
            if (
                    args.training_type == FEDML_TRAINING_PLATFORM_CROSS_DEVICE
                    or args.training_type == FEDML_TRAINING_PLATFORM_CROSS_SILO
            ):
                if args.rank == 0:
                    client_id_list = []
                    for client_idx in range(args.client_num_per_round):
                        client_id_list.append(client_idx + 1)
                    args.client_id_list = str(client_id_list)
                    print("------------------server client_id_list = {}-------------------".format(args.client_id_list))
                else:
                    # for the client, we only specify its client id in the list, not including others.
                    client_id_list = []
                    client_id_list.append(args.rank)
                    args.client_id_list = str(client_id_list)
                    print("------------------client client_id_list = {}-------------------".format(args.client_id_list))
            else:
                print(
                    "training_type != FEDML_TRAINING_PLATFORM_CROSS_DEVICE and training_type != "
                    "FEDML_TRAINING_PLATFORM_CROSS_SILO"
                )
        else:
            print("args.client_id_list is not None")
    else:
        print("using_mlops = true")


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
