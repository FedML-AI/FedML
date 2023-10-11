from .FedAVGAggregator import FedAVGAggregator
from .FedAVGTrainer import FedAVGTrainer
from .FedAvgClientManager import FedAVGClientManager
from .FedAvgServerManager import FedAVGServerManager
from ....core import ClientTrainer, ServerAggregator
from ....core.dp.fedml_differential_privacy import FedMLDifferentialPrivacy
from ....core.security.fedml_attacker import FedMLAttacker
from ....core.security.fedml_defender import FedMLDefender
from ....ml.aggregator.aggregator_creator import create_server_aggregator
from ....ml.trainer.trainer_creator import create_model_trainer


def FedML_FedAvg_distributed(
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
    """
    Run Federated Averaging (FedAvg) in a distributed setting.

    Args:
        args: The command-line arguments and configuration for the FedAvg process.
        process_id (int): The unique identifier for the current process.
        worker_number (int): The total number of worker processes.
        comm: The communication backend for inter-process communication.
        device: The target device (e.g., CPU or GPU) for training.
        dataset: The dataset for training and testing.
        model: The machine learning model to be trained.
        client_trainer (ClientTrainer, optional): The client trainer responsible for local training.
        server_aggregator (ServerAggregator, optional): The server aggregator for model aggregation.
    """
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
    """
    Initialize the server for FedAvg.

    Args:
        args: The command-line arguments and configuration for the FedAvg process.
        device: The target device (e.g., CPU or GPU) for training.
        comm: The communication backend for inter-process communication.
        rank (int): The rank or identifier of the server process.
        size (int): The total number of processes.
        model: The machine learning model to be trained.
        train_data_num (int): The number of training samples.
        train_data_global: The global training dataset.
        test_data_global: The global testing dataset.
        train_data_local_dict: A dictionary mapping client IDs to their local training datasets.
        test_data_local_dict: A dictionary mapping client IDs to their local testing datasets.
        train_data_local_num_dict: A dictionary mapping client IDs to the number of local training samples.
        server_aggregator: The server aggregator for model aggregation.
    """
    if server_aggregator is None:
        server_aggregator = create_server_aggregator(model, args)
    server_aggregator.set_id(-1)

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
        server_aggregator,
    )

    # start the distributed training
    backend = args.backend
    server_manager = FedAVGServerManager(args, aggregator, comm, rank, size, backend)
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
    """
    Initialize a client for FedAvg.

    Args:
        args: The command-line arguments and configuration for the FedAvg process.
        device: The target device (e.g., CPU or GPU) for training.
        comm: The communication backend for inter-process communication.
        process_id (int): The unique identifier for the client process.
        size (int): The total number of processes.
        model: The machine learning model to be trained.
        train_data_num (int): The number of training samples.
        train_data_local_num_dict: A dictionary mapping client IDs to the number of local training samples.
        train_data_local_dict: A dictionary mapping client IDs to their local training datasets.
        test_data_local_dict: A dictionary mapping client IDs to their local testing datasets.
        model_trainer (ModelTrainer, optional): The model trainer responsible for local training.
    """
    client_index = process_id - 1
    if model_trainer is None:
        model_trainer = create_model_trainer(model, args)
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
