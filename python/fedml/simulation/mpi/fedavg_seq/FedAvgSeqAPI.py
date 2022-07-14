from .FedAVGAggregator import FedAVGAggregator
from .FedAVGTrainer import FedAVGTrainer
from .FedAvgClientManager import FedAVGClientManager
from .FedAvgServerManager import FedAVGServerManager
from .my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from .my_model_trainer_nwp import MyModelTrainer as MyModelTrainerNWP
from .my_model_trainer_tag_prediction import MyModelTrainer as MyModelTrainerTAG
from ....core.security.fedml_attacker import FedMLAttacker
from ....core.security.fedml_defender import FedMLDefender


def FedML_FedAvgSeq_distributed(
    args,
    process_id,
    worker_number,
    comm,
    device,
    dataset,
    model,
    model_trainer=None,
    preprocessed_sampling_lists=None,
):
    [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ] = dataset

    FedMLAttacker.get_instance().init(args)
    FedMLDefender.get_instance().init(args)

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
            test_data_local_dict,
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
    aggregator = FedAVGAggregator(
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
    backend = args.backend
    if preprocessed_sampling_lists is None:
        server_manager = FedAVGServerManager(
            args, aggregator, comm, rank, size, backend
        )
    else:
        server_manager = FedAVGServerManager(
            args,
            aggregator,
            comm,
            rank,
            size,
            backend,
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
    test_data_local_dict,
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
    backend = args.backend
    trainer = FedAVGTrainer(
        client_index,
        train_data_local_dict,
        train_data_local_num_dict,
        test_data_local_dict,
        train_data_num,
        device,
        args,
        model_trainer,
    )
    client_manager = FedAVGClientManager(args, trainer, comm, process_id, size, backend)
    client_manager.run()
