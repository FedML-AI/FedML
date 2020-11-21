from mpi4py import MPI

from fedml_api.distributed.fedseg.FedSegAggregator import FedSegAggregator
from fedml_api.distributed.fedseg.FedSegTrainer import FedSegTrainer
from fedml_api.distributed.fedseg.FedSegClientManager import FedSegClientManager
from fedml_api.distributed.fedseg.FedSegServerManager import FedSegServerManager


def FedML_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number


def FedML_FedSeg_distributed(process_id, worker_number, device, comm, model, train_data_num, train_data_global, test_data_global,
                 train_data_local_num_dict, train_data_local_dict, test_data_local_dict, n_class, args):                 
    if process_id == 0:
        init_server(args, device, comm, process_id, worker_number, model, train_data_num, train_data_global,
                    test_data_global, train_data_local_dict, test_data_local_dict, train_data_local_num_dict, n_class)
    else:
        init_client(args, device, comm, process_id, worker_number, model, train_data_num, train_data_local_num_dict,
                    train_data_local_dict)


def init_server(args, device, comm, rank, size, model, train_data_num, train_data_global, test_data_global,
                train_data_local_dict, test_data_local_dict, train_data_local_num_dict, n_class):
    # aggregator
    worker_num = size - 1
    aggregator = FedSegAggregator(train_data_global, test_data_global, train_data_num,
                                  train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_num, device, model, n_class, args)

    # start the distributed training
    server_manager = FedSegServerManager(args, aggregator, comm, rank, size)
    server_manager.send_init_msg()
    server_manager.run()

# TODO List   
# - client should be modified to decentralized worker
# - add group id 
# - Add MPC related setting
def init_client(args, device, comm, process_id, size, model, train_data_num, train_data_local_num_dict, train_data_local_dict):
    # trainer
    client_index = process_id - 1
    trainer = FedSegTrainer(client_index, train_data_local_dict, train_data_local_num_dict, train_data_num, device, model, args)

    client_manager = FedSegClientManager(args, trainer, comm, process_id, size)
    client_manager.run()
