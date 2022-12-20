from .FLTrainer import FLTrainer
from .ClientManager import ClientManager
from .ServerManager import ServerManager

from .train_seq.ClientManager import SeqClientManager
from .train_seq.ServerManager import SeqServerManager

from ....core import ClientTrainer, ServerAggregator
from ....core.dp.fedml_differential_privacy import FedMLDifferentialPrivacy
from ....core.security.fedml_attacker import FedMLAttacker
from ....core.security.fedml_defender import FedMLDefender
from ....ml.aggregator.aggregator_creator import create_server_aggregator
from ....ml.trainer.trainer_creator import create_model_trainer
from .default_aggregator import DefaultServerAggregator

from ....core.compression.fedml_compression import FedMLCompression
from ....core.compression import MLcompression


def FedML_distributed(
    args,
    process_id,
    worker_number,
    comm,
    device,
    dataset,
    model,
    client_trainer: ClientTrainer = None,
    server_aggregator: ServerAggregator = None,
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
    FedMLDifferentialPrivacy.get_instance().init(args)
    FedMLCompression.get_instance("upload").init(args, model)
    FedMLCompression.get_instance("download").init(args, model)

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
            server_aggregator
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
            client_trainer,
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
    server_aggregator
):
    if server_aggregator is None:
        server_aggregator = create_server_aggregator(model, args)
    server_aggregator.set_id(-1)

    # aggregator
    worker_num = size - 1

    # server_aggregator = DefaultServerAggregator(
    #     train_data_global,
    #     test_data_global,
    #     train_data_num,
    #     train_data_local_dict,
    #     test_data_local_dict,
    #     train_data_local_num_dict,
    #     worker_num,
    #     device,
    #     args,
    #     model,
    # )

    # start the distributed training
    backend = args.backend
    if hasattr(args, "hierarchical_agg") and args.hierarchical_agg:
        server_manager = SeqServerManager(args, server_aggregator, comm, rank, size, backend)
    else:
        server_manager = ServerManager(args, server_aggregator, comm, rank, size, backend)
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
        model_trainer = create_model_trainer(model, args)
    model_trainer.set_id(client_index)

    # from fedml.ml.trainer.my_model_trainer_classification_new import ModelTrainerCLS 
    # model_trainer = ModelTrainerCLS(model, args)

    backend = args.backend
    trainer = FLTrainer(
        client_index,
        train_data_local_dict,
        train_data_local_num_dict,
        test_data_local_dict,
        train_data_num,
        device,
        args,
        model_trainer,
    )

    if hasattr(args, "hierarchical_agg") and args.hierarchical_agg:
        client_manager = SeqClientManager(args, trainer, comm, process_id, size, backend)
    else:
        client_manager = ClientManager(args, trainer, comm, process_id, size, backend)
    client_manager.run()
