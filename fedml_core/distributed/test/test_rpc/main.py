import argparse
import logging
import os
import sys
from time import sleep

from mpi4py import MPI

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../../../../")))

from fedml_core.distributed.test.test_rpc.dummy_algorithm.client_manager import RPCClientManager
from fedml_core.distributed.test.test_rpc.dummy_algorithm.server_manager import RPCServerManager


def add_args(parser):
    parser.add_argument("--rank", type=int, default=0)

    parser.add_argument("--backend", type=str, default="GRPC")

    parser.add_argument(
        "--enable_cuda_rpc",
        default=False,
        action="store_true",
        help="Enable cuda rpc (only for TRPC backend)",
    )

    parser.add_argument(
        "--gpu_mapping_file",
        type=str,
        default="gpu_mapping.yaml",
        help="the gpu utilization file for servers and clients. If there is no \
                        gpu_util_file, gpu will not be used.",
    )

    parser.add_argument(
        "--gpu_mapping_key", type=str, default="mapping_default", help="the key in gpu utilization file"
    )

    parser.add_argument(
        "--grpc_ipconfig_path",
        type=str,
        default="grpc_ipconfig.csv",
        help="config table containing ipv4 address of grpc server",
    )

    parser.add_argument(
        "--trpc_master_config_path",
        type=str,
        default="trpc_master_config.csv",
        help="config indicating ip address and port of the master (rank 0) node",
    )

    args = parser.parse_args()
    return args


def run_worker(args, rank, size):
    if rank == 0:
        server_manager = RPCServerManager(args, rank=rank, size=size, backend=args.backend)
        sleep(1)
        server_manager.send_model_params()
        server_manager.run()
    else:

        client_manager = RPCClientManager(args, rank=rank, size=size, backend=args.backend)
        client_manager.run()


if __name__ == "__main__":
    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    rank = args.rank
    world_size = 2

    # customize the log format
    # logging.basicConfig(level=logging.INFO,
    logging.basicConfig(
        level=logging.DEBUG,
        format="rank - "
        + str(rank)
        + " - %(asctime)s.%(msecs)03d %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d,%H:%M:%S",
    )

    logging.info(args)

    run_worker(args, rank, world_size)
