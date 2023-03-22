from mpi4py import MPI
from torch import nn

from .client import SplitNN_client
from .client_manager import SplitNNClientManager
from .server import SplitNN_server
from .server_manager import SplitNNServerManager


def SplitNN_distributed(
    process_id, worker_number, device, comm, model, dataset, args,
):
    [
        train_data_num,
        local_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local,
        test_data_local,
        class_num,
    ] = dataset

    fc_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Flatten(), nn.Linear(fc_features, class_num))
    split_layer = 1
    # Split The model
    client_model = nn.Sequential(*nn.ModuleList(model.children())[:split_layer])
    server_model = nn.Sequential(*nn.ModuleList(model.children())[split_layer:])

    server_rank = 0
    if process_id == server_rank:
        init_server(comm, server_model, process_id, worker_number, device, args)
    else:
        init_client(
            comm,
            client_model,
            worker_number,
            train_data_local,
            test_data_local,
            process_id,
            server_rank,
            args.epochs,
            device,
            args,
        )


def init_server(comm, server_model, process_id, worker_number, device, args):
    arg_dict = {
        "comm": comm,
        "model": server_model,
        "max_rank": worker_number - 1,
        "rank": process_id,
        "device": device,
        "args": args,
    }
    server = SplitNN_server(arg_dict)
    server_manager = SplitNNServerManager(arg_dict, server)
    server_manager.run()


def init_client(
    comm, client_model, worker_number, train_data_local, test_data_local, process_id, server_rank, epochs, device, args,
):
    client_ID = process_id - 1
    arg_dict = {
        "client_index": client_ID,
        "comm": comm,
        "trainloader": train_data_local,
        "testloader": test_data_local,
        "model": client_model,
        "rank": process_id,
        "server_rank": server_rank,
        "max_rank": worker_number - 1,
        "epochs": epochs,
        "device": device,
        "args": args,
    }
    client = SplitNN_client(arg_dict)
    client_manager = SplitNNClientManager(arg_dict, client)
    client_manager.run()
