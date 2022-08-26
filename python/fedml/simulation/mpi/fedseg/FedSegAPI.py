import logging

from mpi4py import MPI

from .FedSegAggregator import FedSegAggregator
from .FedSegClientManager import FedSegClientManager
from .FedSegServerManager import FedSegServerManager
from .FedSegTrainer import FedSegTrainer
from .MyModelTrainer import MyModelTrainer


def FedML_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number


def FedML_FedSeg_distributed(
    process_id,
    worker_number,
    device,
    comm,
    model,
    train_data_num,
    train_data_local_num_dict,
    train_data_local_dict,
    test_data_local_dict,
    args,
    model_trainer=None,
):

    if process_id == 0:
        init_server(args, device, comm, process_id, worker_number, model, model_trainer)
    else:
        init_client(
            args,
            device,
            comm,
            process_id,
            worker_number,
            model,
            train_data_num,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            model_trainer,
        )


def init_server(args, device, comm, rank, size, model, model_trainer):
    logging.info("Initializing Server")

    if model_trainer is None:
        model_trainer = MyModelTrainer(model, args)
    model_trainer.set_id(-1)

    # aggregator
    worker_num = size - 1
    aggregator = FedSegAggregator(worker_num, device, model, args, model_trainer)

    # start the distributed training
    server_manager = FedSegServerManager(args, aggregator, comm, rank, size)
    server_manager.send_init_msg()
    server_manager.run()


def init_client(
    args,
    device,
    comm,
    process_id,
    size,
    model,
    train_data_num,
    train_data_local_num_dict,
    train_data_local_dict,
    test_data_local_dict,
    model_trainer,
):

    client_index = process_id - 1
    logging.info("Initializing Client: {0}".format(client_index))

    if model_trainer is None:
        model_trainer = MyModelTrainer(model, args)
    model_trainer.set_id(client_index)

    # trainer
    trainer = FedSegTrainer(
        client_index,
        train_data_local_dict,
        train_data_local_num_dict,
        train_data_num,
        test_data_local_dict,
        device,
        model,
        args,
        model_trainer,
    )
    client_manager = FedSegClientManager(args, trainer, comm, process_id, size)
    client_manager.run()
