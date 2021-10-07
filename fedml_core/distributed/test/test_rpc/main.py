import argparse
import logging
import os
import sys

from mpi4py import MPI

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../../FedML")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../FedML")))

from fedml_core.distributed.test.test_rpc.dummy_algorithm.client_manager import RPCClientManager
from fedml_core.distributed.test.test_rpc.dummy_algorithm.server_manager import RPCServerManager


def add_args(parser):

    parser.add_argument("--backend", type=str, default="GRPC")

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
        client_manager = RPCClientManager(args, rank=rank, size=size, backend=args.backend)
        client_manager.run()
    else:
        server_manager = RPCServerManager(args, rank=rank, size=size, backend=args.backend)
        server_manager.send_model_params()
        server_manager.run()


if __name__ == "__main__":

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    # customize the log format
    # logging.basicConfig(level=logging.INFO,
    logging.basicConfig(
        level=logging.DEBUG,
        format=" - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
        datefmt="%a, %d %b %Y %H:%M:%S",
    )

    logging.info(args)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = 2
    run_worker(args, rank, world_size)
