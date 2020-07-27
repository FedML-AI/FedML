from mpi4py import MPI

from fedml_api.distributed.split_nn.server import SplitNN_server
from fedml_api.distributed.split_nn.client import SplitNN_client

def SplitNN_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number

def SplitNN_distributed(process_id, worker_number, device, comm, client_model,
                        server_model, train_data_num, train_data_global, test_data_global,
                        local_data_num, train_data_local, test_data_local, args):
    if process_id == 0:
        init_server(comm, server_model, worker_number, device)
    else:
        server_rank = 0
        init_client(comm, client_model, worker_number, train_data_local, test_data_local,
                    process_id, server_rank, args.epochs, device)

def init_server(comm, server_model, worker_number, device):
    arg_dict = {"comm": comm, "model": server_model, "max_rank": worker_number - 1,
                "device": device}
    server = SplitNN_server(arg_dict)
    server.run()

def init_client(comm, client_model, worker_number, train_data_local, test_data_local,
                process_id, server_rank, epochs, device):
    arg_dict = {"comm": comm, "trainloader": train_data_local, "testloader": test_data_local,
                "model": client_model, "rank": process_id, "server_rank": server_rank,
                "max_rank": worker_number - 1, "epochs": epochs, "device": device}
    client = SplitNN_client(arg_dict)
    client.run()
