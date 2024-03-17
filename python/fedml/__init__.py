import logging
from copy import deepcopy

import multiprocess as multiprocessing
import os
import random

import numpy as np
import torch

import fedml
from .computing.scheduler.env.collect_env import collect_env
from .constants import (
    FEDML_BACKEND_SERVICE_URL_DEV,
    FEDML_BACKEND_SERVICE_URL_LOCAL,
    FEDML_BACKEND_SERVICE_URL_RELEASE,
    FEDML_BACKEND_SERVICE_URL_TEST,
    FEDML_MQTT_DOMAIN_DEV,
    FEDML_MQTT_DOMAIN_LOCAL,
    FEDML_MQTT_DOMAIN_TEST,
    FEDML_MQTT_DOMAIN_RELEASE,
    FEDML_S3_DOMAIN_LOCAL,
    FEDML_TRAINING_PLATFORM_SIMULATION,
    FEDML_SIMULATION_TYPE_SP,
    FEDML_SIMULATION_TYPE_MPI,
    FEDML_SIMULATION_TYPE_NCCL,
    FEDML_TRAINING_PLATFORM_CROSS_SILO,
    FEDML_TRAINING_PLATFORM_CROSS_DEVICE,
    FEDML_TRAINING_PLATFORM_CROSS_CLOUD,
    FEDML_TRAINING_PLATFORM_SERVING,
)
from .core.common.ml_engine_backend import MLEngineBackend

_global_training_type = None
_global_comm_backend = None

__version__ = "0.8.28"


# This is the deployment environment used for different roles (RD/PM/BD/Public Developers). Potential VALUE: local, dev, test, release
# three ways to set _global_env_version:
# 1. set the Linux environment variable FEDML_ENV_VERSION by calling: 
#                   export FEDML_ENV_VERSION=VALUE
#         (this is typically used when a process (e.g., agent) calling antoher process (e.g., job))
# 2. by API python API: 
#                   import fedml
#                   fedml.set_env_version(VALUE)
#         (this is typically used when you want to control versions in your python scripts)
# 3. by CLI: 
#                   fedml launch job.yaml -v local
#         (this is typically used when you use CLIs)
# 4. by arguments in yaml file passed into python program. For historical reasons, we support two arguments
#             job.yaml:
#                   config_version: VALUE
#                   env_version: VALUE
#        (this is typically used when you develop your own ML training/deployment job by using FedML framework)
# to be consistant across all geo-distributed processes in an entire job execution 
# and make the all components inside FedML library are aligned with the same environment, we use the following policy:
# 1) no matter how we set _global_env_version, we always save its value into FEDML_ENV_VERSION and read its value for the entire source code.
# 2) if both 1/2/3 and 4 are set, we let the value in 1/2/3 overides 4 (to make sure the agent and the job are aligned with the same environment)
# 3) if _global_env_version is not set, we make it as "release"


def init(args=None, check_env=True, should_init_logs=True):
    if args is None:
        args = load_arguments(fedml._global_training_type, fedml._global_comm_backend)

    """Initialize FedML Engine."""
    # only when the env version is None, we refer to the python configuration arguments. 
    if get_env_version() is None:
        if hasattr(args, "config_version") and args.config_version is not None:
            set_env_version(args.config_version)
            delattr(args, "config_version")
        elif hasattr(args, "env_version") and args.env_version is not None:
            set_env_version(args.env_version)
            delattr(args, "env_version")
        else:
            set_env_version("release")
    # after environment is set, only fedml.get_env_version() is used to get the environment version

    if check_env and hasattr(args, "training_type") and args.training_type != fedml.FEDML_TRAINING_PLATFORM_SIMULATION:
        collect_env()

    if hasattr(args, "training_type"):
        fedml._global_training_type = args.training_type
    if hasattr(args, "backend"):
        fedml._global_comm_backend = args.backend

    """
    # Windows/Linux/MacOS compatability issues on multi-processing
    # https://github.com/pytorch/pytorch/issues/3492
    """
    if multiprocessing.get_start_method() != "spawn":
        # force all platforms (Windows/Linux/MacOS) to use the same way (spawn) for multiprocessing
        multiprocessing.set_start_method("spawn", force=True)

    """
    # https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    seed = args.random_seed if hasattr(args, "random_seed") else 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    mlops.pre_setup(args)

    if not hasattr(args, "training_type"):
        setattr(args, "training_type", fedml._global_training_type)

    if not hasattr(args, "backend"):
        setattr(args, "backend", fedml._global_comm_backend)

    if hasattr(args, "training_type"):
        if args.training_type == FEDML_TRAINING_PLATFORM_SIMULATION and hasattr(args,
                                                                                "backend") and args.backend == "MPI":
            args = _init_simulation_mpi(args)

        elif args.training_type == FEDML_TRAINING_PLATFORM_SIMULATION and hasattr(args,
                                                                                  "backend") and args.backend == "sp":
            args = _init_simulation_sp(args)
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
                _init_cross_silo_horizontal(args)
            elif args.scenario == "hierarchical":
                args = _init_cross_silo_hierarchical(args)

        elif args.training_type == FEDML_TRAINING_PLATFORM_CROSS_DEVICE:
            args = _init_cross_device(args)
        elif args.training_type == FEDML_TRAINING_PLATFORM_CROSS_CLOUD:
            args = _init_cross_cloud(args)
        elif args.training_type == FEDML_TRAINING_PLATFORM_SERVING:
            args = _init_model_serving(args)
        else:
            raise Exception(
                "no such setting: training_type = {}, backend = {}".format(args.training_type, args.backend))

    _manage_profiling_args(args)

    _update_client_id_list(args)

    mlops.init(args, should_init_logs=should_init_logs)

    if hasattr(args, "rank") and hasattr(args, "worker_num"):
        if hasattr(args, "process_id") and args.process_id is not None:
            logging.info("args.rank = {}, args.process_id = {}, args.worker_num = {}".format(args.rank, args.process_id,
                                                                                             args.worker_num))
        else:
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


def _init_simulation_mpi(args):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    world_size = comm.Get_size()
    args.comm = comm
    args.process_id = process_id
    args.rank = process_id
    args.worker_num = world_size
    if process_id == 0:
        args.role = "server"
    else:
        args.role = "client"
    return args


def _init_simulation_sp(args):
    return args


def _init_simulation_nccl(args):
    return


def _manage_profiling_args(args):
    if not hasattr(args, "sys_perf_profiling"):
        args.sys_perf_profiling = True
    if not hasattr(args, "sys_perf_profiling"):
        args.sys_perf_profiling = True

    if hasattr(args, "sys_perf_profiling") and args.sys_perf_profiling:
        from .core.mlops.mlops_profiler_event import MLOpsProfilerEvent

        MLOpsProfilerEvent.enable_sys_perf_profiling()

    if hasattr(args, "enable_wandb") and args.enable_wandb:
        wandb_only_server = getattr(args, "wandb_only_server", None)
        if (wandb_only_server and args.rank == 0 and args.process_id == 0) or not wandb_only_server:
            wandb_entity = getattr(args, "wandb_entity", None)
            if wandb_entity is not None:
                wandb_args = {
                    "entity": args.wandb_entity,
                    "project": args.wandb_project,
                    "config": args,
                }
            else:
                wandb_args = {
                    "project": args.wandb_project,
                    "config": args,
                }

            if hasattr(args, "run_name"):
                wandb_args["name"] = args.run_name

            if hasattr(args, "wandb_group_id"):
                # wandb_args["group"] = args.wandb_group_id
                wandb_args["group"] = "Test1"
                wandb_args["name"] = f"Client {args.rank}"
                wandb_args["job_type"] = str(args.rank)

            import wandb

            wandb.init(**wandb_args)

            from .core.mlops.mlops_profiler_event import MLOpsProfilerEvent

            MLOpsProfilerEvent.enable_wandb_tracking()


def _manage_cuda_rpc_args(args):
    if (not hasattr(args, "enable_cuda_rpc")) or (not args.using_gpu):
        args.enable_cuda_rpc = False

    if args.enable_cuda_rpc and args.backend != "TRPC":
        args.enable_cuda_rpc = False
        print("Argument enable_cuda_rpc is ignored. Cuda RPC only works with TRPC backend.")

    # When Cuda RPC is not used, tensors should be moved to cpu before transfer with TRPC
    if (not args.enable_cuda_rpc) and args.backend == "TRPC":
        args.cpu_transfer = True
    else:
        args.cpu_transfer = False

    # Valudate arguments related to cuda rpc
    if args.enable_cuda_rpc:
        if not hasattr(args, "cuda_rpc_gpu_mapping"):
            raise Exception("Invalid config. cuda_rpc_gpu_mapping is required when enable_cuda_rpc=True")
        assert type(args.cuda_rpc_gpu_mapping) is dict, "Invalid cuda_rpc_gpu_mapping type. Expected dict"
        assert (
                len(args.cuda_rpc_gpu_mapping) == args.worker_num + 1
        ), f"Invalid cuda_rpc_gpu_mapping. Expected list of size {args.worker_num + 1}"

    # print(f"cpu_transfer: {args.cpu_transfer}")
    # print(f"enable_cuda_rpc: {args.enable_cuda_rpc}")


def _manage_mpi_args(args):
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
        # args.worker_num = worker_num
        assert args.worker_num + 1 == world_size, f"Invalid number of mpi processes. Expected {args.worker_num + 1}"
    else:
        args.comm = None


def _init_cross_silo_horizontal(args):
    args.n_proc_in_silo = 1
    args.proc_rank_in_silo = 0
    _manage_mpi_args(args)
    _manage_cuda_rpc_args(args)
    args.process_id = args.rank
    return args


def _init_cross_silo_hierarchical(args):
    _manage_mpi_args(args)
    _manage_cuda_rpc_args(args)

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

        # Launcher Rendezvous
        if not hasattr(args, "launcher_rdzv_port"):
            args.launcher_rdzv_port = 29400

        if not hasattr(args, "n_node_in_silo"):
            args.n_node_in_silo = 1
        if not (hasattr(args, "n_proc_per_node") and args.n_proc_per_node):
            if args.n_node_in_silo == 1 and torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                # Checking if launcher is has spawned enough processes.
                if gpu_count == args.n_proc_in_silo:
                    print(f"Auto assigning GPU to processes.")
                    args.gpu_id = args.proc_rank_in_silo
                else:
                    args.n_proc_per_node = 1
            else:
                args.n_proc_per_node = 1

    print("\nargs.rank = {}, args.n_proc_in_silo: {}".format(args.rank, args.n_proc_in_silo))
    print("args.rank = {}, n_proc_in_silo: {}".format(args.rank, args.n_proc_in_silo))
    print("args.rank = {}, rank_in_node: {}".format(args.rank, args.rank_in_node))
    print("args.rank = {}, proc_rank_in_silo: {}".format(args.rank, args.proc_rank_in_silo))
    exit()
    return args


def _init_cross_cloud(args):
    args.n_proc_in_silo = 1
    args.proc_rank_in_silo = 0
    _manage_mpi_args(args)
    _manage_cuda_rpc_args(args)
    args.process_id = args.rank
    return args


def _init_model_serving(args):
    args.n_proc_in_silo = 1
    args.proc_rank_in_silo = 0
    _manage_cuda_rpc_args(args)
    args.process_id = args.rank
    return args


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
            print("args.client_id_list is not None")


def _init_cross_device(args):
    args.rank = 0  # only server runs on Python package
    args.role = "server"
    return args


def _run_distributed():
    pass


def set_env_version(version):
    os.environ['FEDML_ENV_VERSION'] = version


def get_env_version():
    return "release" if os.environ.get('FEDML_ENV_VERSION') is None else os.environ['FEDML_ENV_VERSION']


def _get_backend_service():
    version = get_env_version()
    # from inspect import getframeinfo, stack
    # caller = getframeinfo(stack()[1][0])    
    # print(f"{caller.filename}:{caller.lineno} - _get_backend_service. version = {version}")
    if version == "local":
        port = int(get_local_on_premise_platform_port())
        if port == 80:
            return f"http://{get_local_on_premise_platform_host()}"
        elif port == 443:
            return f"https://{get_local_on_premise_platform_host()}"
        else:
            return f"http://{get_local_on_premise_platform_host()}:{port}"
    elif version == "dev":
        return FEDML_BACKEND_SERVICE_URL_DEV
    elif version == "test":
        return FEDML_BACKEND_SERVICE_URL_TEST
    else:
        return FEDML_BACKEND_SERVICE_URL_RELEASE


def _get_mqtt_service():
    version = get_env_version()
    # from inspect import getframeinfo, stack
    # caller = getframeinfo(stack()[1][0])    
    # print(f"{caller.filename}:{caller.lineno} - _get_backend_service. version = {version}")
    if version == "local":
        return FEDML_MQTT_DOMAIN_LOCAL
    if version == "dev":
        return FEDML_MQTT_DOMAIN_DEV
    elif version == "test":
        return FEDML_MQTT_DOMAIN_TEST
    else:
        return FEDML_MQTT_DOMAIN_RELEASE


def set_local_on_premise_platform_host(local_on_premise_platform_host):
    os.environ['FEDML_ENV_LOCAL_ON_PREMISE_PLATFORM_HOST'] = local_on_premise_platform_host


def get_local_on_premise_platform_host():
    return os.environ.get('FEDML_ENV_LOCAL_ON_PREMISE_PLATFORM_HOST', "127.0.0.1")


def set_local_on_premise_platform_port(local_on_premise_platform_port):
    os.environ['FEDML_ENV_LOCAL_ON_PREMISE_PLATFORM_PORT'] = str(local_on_premise_platform_port)


def get_local_on_premise_platform_port():
    return os.environ.get('FEDML_ENV_LOCAL_ON_PREMISE_PLATFORM_PORT', 80)


def _get_local_s3_like_service_url():
    return FEDML_S3_DOMAIN_LOCAL
    

from fedml import device
from fedml import data
from fedml import model

from .arguments import load_arguments

from .launch_simulation import run_simulation

from .launch_cross_silo_horizontal import run_cross_silo_server
from .launch_cross_silo_horizontal import run_cross_silo_client

from .launch_cross_silo_hi import run_hierarchical_cross_silo_server
from .launch_cross_silo_hi import run_hierarchical_cross_silo_client

from .launch_cross_cloud import run_cross_cloud_server
from .launch_cross_cloud import run_cross_cloud_client

from .launch_serving import run_model_serving_client
from .launch_serving import run_model_serving_server

from .launch_cross_device import run_mnn_server

from .core.common.ml_engine_backend import MLEngineBackend

from .runner import FedMLRunner

from fedml import api

from fedml import mlops
from fedml.mlops import log, Artifact, log_artifact, log_model, log_metric, log_llm_record, log_endpoint

__all__ = [
    "FedMLRunner",
    "api",
    "log",
    "Artifact",
    "log_artifact",
    "log_model",
    "log_metric",
    "log_llm_record",
    "log_endpoint",
    "mlops",
    "device",
    "data",
    "model",
    "run_simulation",
    "run_cross_silo_server",
    "run_cross_silo_client",
    "run_hierarchical_cross_silo_server",
    "run_hierarchical_cross_silo_client",
    "run_cross_cloud_server",
    "run_cross_cloud_client",
    "run_model_serving_client",
    "run_model_serving_server",
    "run_mnn_server"
]
