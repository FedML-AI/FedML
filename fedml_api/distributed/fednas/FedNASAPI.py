from mpi4py import MPI

from fedml_api.distributed.fednas.FedNASAggregator import FedNASAggregator
from fedml_api.distributed.fednas.FedNASClientManager import FedNASClientManager
from fedml_api.distributed.fednas.FedNASServerManager import FedNASServerManager
from fedml_api.distributed.fednas.FedNASTrainer import FedNASTrainer


def FedML_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number


def FedML_FedNAS_distributed(process_id, worker_number, device, comm, model, train_data_num, train_data_global, test_data_global,
                 local_data_num, train_data_local, test_data_local, args):
    if process_id == 0:
        init_server(args, device, comm, process_id, worker_number, model, train_data_num, train_data_global,
                    test_data_global)
    else:
        init_client(args, device, comm, process_id, worker_number, model, train_data_num, local_data_num,
                    train_data_local, test_data_local)


def init_server(args, device, comm, process_id, worker_number, model, train_data_num, train_data_global, test_data_global):
    # aggregator
    client_num = worker_number - 1
    aggregator = FedNASAggregator(train_data_global, test_data_global, train_data_num, client_num, model, device, args)

    # start the distributed training
    server_manager = FedNASServerManager(args, comm, process_id, worker_number, aggregator)
    server_manager.run()


def init_client(args, device, comm, process_id, worker_number, model, train_data_num, local_data_num, train_data_local, test_data_local):
    # trainer
    client_ID = process_id - 1
    trainer = FedNASTrainer(client_ID, train_data_local, test_data_local, local_data_num, train_data_num, model, device, args)

    client_manager = FedNASClientManager(args, comm, process_id, worker_number, trainer)
    client_manager.run()
