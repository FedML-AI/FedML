from mpi4py import MPI

from fedml_api.distributed.fedavg.FedAVGAggregator import FedAVGAggregator
from fedml_api.distributed.fedavg.FedAVGTrainer import FedAVGTrainer
from fedml_api.distributed.fedavg.FedAvgClientManager import FedAVGClientManager
from fedml_api.distributed.fedavg.FedAvgServerManager import FedAVGServerManager


def FedML_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number


def FedML_FedAvg_distributed(process_id, worker_number, device, comm, model, train_data_num, train_data_global, test_data_global,
                 local_data_num, train_data_local, test_data_local, args):
    if process_id == 0:
        init_server(args, device, comm, process_id, worker_number, model, train_data_num, train_data_global,
                    test_data_global)
    else:
        init_client(args, device, comm, process_id, worker_number, model, train_data_num, local_data_num,
                    train_data_local)


def init_server(args, device, comm, rank, size, model, train_data_num, train_data_global, test_data_global):
    # aggregator
    client_num = size - 1
    aggregator = FedAVGAggregator(train_data_global, test_data_global, train_data_num, client_num, device, model, args)

    # start the distributed training
    server_manager = FedAVGServerManager(args, comm, rank, size, aggregator)
    server_manager.run()


def init_client(args, device, comm, process_id, size, model, train_data_num, local_data_num, train_data_local):
    # trainer
    client_ID = process_id - 1
    trainer = FedAVGTrainer(client_ID, train_data_local, local_data_num, train_data_num, device, model, args)

    client_manager = FedAVGClientManager(args, comm, process_id, size, trainer)
    client_manager.run()
