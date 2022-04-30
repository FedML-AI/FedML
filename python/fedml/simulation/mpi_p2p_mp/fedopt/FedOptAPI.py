from mpi4py import MPI

from .FedOptAggregator import FedOptAggregator
from .FedOptClientManager import FedOptClientManager
from .FedOptServerManager import FedOptServerManager
from .FedOptTrainer import FedOptTrainer
from .my_model_trainer_classification import (
    MyModelTrainer as MyModelTrainerCLS,
)
from .my_model_trainer_nwp import (
    MyModelTrainer as MyModelTrainerNWP,
)
from .my_model_trainer_tag_prediction import (
    MyModelTrainer as MyModelTrainerTAG,
)


def FedML_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number


def FedML_FedOpt_distributed(
    process_id,
    worker_number,
    device,
    comm,
    model,
    train_data_num,
    train_data_global,
    test_data_global,
    train_data_local_num_dict,
    train_data_local_dict,
    test_data_local_dict,
    args,
    model_trainer=None,
    preprocessed_sampling_lists=None,
):
    if process_id == 0:
        init_server(
            args,
            device,
            comm,
            process_id,
            worker_number,
            model,
            train_data_num,
            train_data_global,
            test_data_global,
            train_data_local_dict,
            test_data_local_dict,
            train_data_local_num_dict,
            model_trainer,
            preprocessed_sampling_lists,
        )
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
            model_trainer,
        )


def init_server(
    args,
    device,
    comm,
    rank,
    size,
    model,
    train_data_num,
    train_data_global,
    test_data_global,
    train_data_local_dict,
    test_data_local_dict,
    train_data_local_num_dict,
    model_trainer,
    preprocessed_sampling_lists=None,
):
    if model_trainer is None:
        if args.dataset == "stackoverflow_lr":
            model_trainer = MyModelTrainerTAG(model)
        elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp"]:
            model_trainer = MyModelTrainerNWP(model)
        else:  # default model trainer is for classification problem
            model_trainer = MyModelTrainerCLS(model)
    model_trainer.set_id(-1)
    # aggregator
    worker_num = size - 1
    aggregator = FedOptAggregator(
        train_data_global,
        test_data_global,
        train_data_num,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict,
        worker_num,
        device,
        args,
        model_trainer,
    )

    # start the distributed training
    if preprocessed_sampling_lists is None:
        server_manager = FedOptServerManager(args, aggregator, comm, rank, size)
    else:
        server_manager = FedOptServerManager(
            args,
            aggregator,
            comm,
            rank,
            size,
            backend="MPI",
            is_preprocessed=True,
            preprocessed_client_lists=preprocessed_sampling_lists,
        )
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
    model_trainer=None,
):
    client_index = process_id - 1
    if model_trainer is None:
        if args.dataset == "stackoverflow_lr":
            model_trainer = MyModelTrainerTAG(model)
        elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp"]:
            model_trainer = MyModelTrainerNWP(model)
        else:  # default model trainer is for classification problem
            model_trainer = MyModelTrainerCLS(model)
    model_trainer.set_id(client_index)

    trainer = FedOptTrainer(
        client_index,
        train_data_local_dict,
        train_data_local_num_dict,
        train_data_num,
        device,
        args,
        model_trainer,
    )
    client_manager = FedOptClientManager(args, trainer, comm, process_id, size)
    client_manager.run()
