from mpi4py import MPI

from fedml_api.distributed.fedavg_robust.FedAvgRobustAggregator import FedAvgRobustAggregator
from fedml_api.distributed.fedavg_robust.FedAvgRobustTrainer import FedAvgRobustTrainer
from fedml_api.distributed.fedavg_robust.FedAvgRobustClientManager import FedAvgRobustClientManager
from fedml_api.distributed.fedavg_robust.FedAvgRobustServerManager import FedAvgRobustServerManager


def FedML_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number


def FedML_FedAvgRobust_distributed(process_id, worker_number, device, comm, model, train_data_num, train_data_global, test_data_global,
                 train_data_local_num_dict, train_data_local_dict, test_data_local_dict, 
                 poisoned_train_loader, targetted_task_test_loader, num_dps_poisoned_dataset, args):
    if process_id == 0:
        init_server(args, device, comm, process_id, worker_number, model, train_data_num, train_data_global,
                    test_data_global, train_data_local_dict, test_data_local_dict, train_data_local_num_dict, targetted_task_test_loader)
    else:
        init_client(args, device, comm, process_id, worker_number, model, train_data_num, train_data_local_num_dict,
                    train_data_local_dict, poisoned_train_loader, num_dps_poisoned_dataset)


def init_server(args, device, comm, rank, size, model, train_data_num, train_data_global, test_data_global,
                train_data_local_dict, test_data_local_dict, train_data_local_num_dict, targetted_task_test_loader):
    # aggregator
    worker_num = size - 1
    aggregator = FedAvgRobustAggregator(train_data_global, test_data_global, train_data_num,
                                  train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_num, device, model, 
                                  targetted_task_test_loader, args)

    # start the distributed training
    server_manager = FedAvgRobustServerManager(args, aggregator, comm, rank, size)
    server_manager.send_init_msg()
    server_manager.run()


def init_client(args, device, comm, process_id, size, model, train_data_num, train_data_local_num_dict, train_data_local_dict, poisoned_train_loader, num_dps_poisoned_dataset):
    # trainer
    client_index = process_id - 1
    trainer = FedAvgRobustTrainer(client_index, train_data_local_dict, train_data_local_num_dict, train_data_num, device, model, poisoned_train_loader, num_dps_poisoned_dataset, args)

    client_manager = FedAvgRobustClientManager(args, trainer, comm, process_id, size)
    client_manager.run()