from mpi4py import MPI

from fedml_api.distributed.turboaggregate.TA_Aggregator import FedAVGAggregator
from fedml_api.distributed.turboaggregate.TA_Trainer import FedAVGTrainer
from fedml_api.distributed.turboaggregate.TA_decentralized_worker_manager import TA_ClientManager



def FedML_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number


def FedML_FedAvg_distributed(process_id, worker_number, device, comm, model, train_data_num, train_data_global, test_data_global,
                 train_data_local_num_dict, train_data_local_dict, test_data_local_dict, args):
    if process_id == 0:
        init_server(args, device, comm, process_id, worker_number, model, train_data_num, train_data_global,
                    test_data_global, train_data_local_dict, test_data_local_dict, train_data_local_num_dict)
    else:
        init_client(args, device, comm, process_id, worker_number, model, train_data_num, train_data_local_num_dict,
                    train_data_local_dict)


def init_server(args, device, comm, rank, size, model, train_data_num, train_data_global, test_data_global,
                train_data_local_dict, test_data_local_dict, train_data_local_num_dict):
    # aggregator
    worker_num = size - 1
    aggregator = FedAVGAggregator(train_data_global, test_data_global, train_data_num,
                                  train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_num, device, model, args)

    # start the distributed training
    server_manager = FedAVGServerManager(args, aggregator, comm, rank, size)
    server_manager.send_init_msg()
    server_manager.run()


def init_client(args, device, comm, process_id, size, model, train_data_num, train_data_local_num_dict, train_data_local_dict):
    # trainer
    client_index = process_id - 1
    trainer = FedAVGTrainer(client_index, train_data_local_dict, train_data_local_num_dict, train_data_num, device, model, args)

    client_manager = FedAVGClientManager(args, trainer, comm, process_id, size)
    client_manager.run()
