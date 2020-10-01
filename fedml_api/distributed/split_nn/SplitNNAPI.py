from mpi4py import MPI

from fedml_api.distributed.split_nn.client import SplitNN_client
from fedml_api.distributed.split_nn.client_manager import SplitNNClientManager
from fedml_api.distributed.split_nn.server import SplitNN_server
from fedml_api.distributed.split_nn.server_manager import SplitNNServerManager


def SplitNN_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number

def SplitNN_distributed(process_id, worker_number, device, comm, client_model,
                        server_model, train_data_num, train_data_global, test_data_global,
                        local_data_num, train_data_local, test_data_local, args):
    server_rank = 0
    if process_id == server_rank:
        init_server(comm, server_model, process_id, worker_number, device, args)
    else:
        init_client(comm, client_model, worker_number, train_data_local, test_data_local,
                    process_id, server_rank, args.epochs, device, args)

def init_server(comm, server_model, process_id, worker_number, device, args):
    arg_dict = {"comm": comm, "model": server_model, "max_rank": worker_number - 1,
                "rank": process_id, "device": device, "args": args}
    server = SplitNN_server(arg_dict)
    server_manager = SplitNNServerManager(arg_dict, server)
    server_manager.run()

def init_client(comm, client_model, worker_number, train_data_local, test_data_local,
                process_id, server_rank, epochs, device, args):
    arg_dict = {"comm": comm, "trainloader": train_data_local, "testloader": test_data_local,
                "model": client_model, "rank": process_id, "server_rank": server_rank,
                "max_rank": worker_number - 1, "epochs": epochs, "device": device, "args": args}
    client = SplitNN_client(arg_dict)
    client_manager = SplitNNClientManager(arg_dict, client)
    client_manager.run()
