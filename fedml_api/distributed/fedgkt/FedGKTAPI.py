from mpi4py import MPI

from fedml_api.distributed.fedgkt.GKTClientManager import GKTClientMananger
from fedml_api.distributed.fedgkt.GKTClientTrainer import GKTClientTrainer
from fedml_api.distributed.fedgkt.GKTServerManager import GKTServerMananger
from fedml_api.distributed.fedgkt.GKTServerTrainer import GKTServerTrainer


def FedML_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number


def FedML_FedGKT_distributed(process_id, worker_number, device, comm, model, train_data_local_num_dict,
                             train_data_local_dict, test_data_local_dict, args):
    if process_id == 0:
        init_server(args, device, comm, process_id, worker_number, model)
    else:
        init_client(args, device, comm, process_id, worker_number, model, train_data_local_dict,
                    test_data_local_dict, train_data_local_num_dict)


def init_server(args, device, comm, rank, size, model):

    # aggregator
    client_num = size - 1
    server_trainer = GKTServerTrainer(client_num, device, model, args)

    # start the distributed training
    server_manager = GKTServerMananger(args, server_trainer, comm, rank, size)
    server_manager.run()


def init_client(args, device, comm, process_id, size, model, train_data_local_dict, test_data_local_dict,
                train_data_local_num_dict):
    client_ID = process_id - 1

    # 2. initialize the trainer
    trainer = GKTClientTrainer(client_ID, train_data_local_dict, test_data_local_dict, train_data_local_num_dict,
                               device, model, args)

    # 3. start the distributed training
    client_manager = GKTClientMananger(args, trainer, comm, process_id, size)
    client_manager.run()