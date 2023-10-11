from .AsyncFedAVGAggregator import AsyncFedAVGAggregator
from .AsyncFedAVGTrainer import AsyncFedAVGTrainer
from .AsyncFedAvgClientManager import AsyncFedAVGClientManager
from .AsyncFedAvgServerManager import AsyncFedAVGServerManager
from ....core.dp.fedml_differential_privacy import FedMLDifferentialPrivacy
from ....core.security.fedml_attacker import FedMLAttacker
from ....core.security.fedml_defender import FedMLDefender
from ....ml.trainer.trainer_creator import create_model_trainer


def FedML_Async_distributed(
    args, process_id, worker_number, comm, device, dataset, model, model_trainer=None, preprocessed_sampling_lists=None,
):
    """
    Run the asynchronous federated learning process.

    Args:
        args (object): An object containing the configuration parameters.
        process_id (int): The unique ID of the current process.
        worker_number (int): The total number of worker processes.
        comm (object): The communication object.
        device (object): The device to run the training on (e.g., GPU).
        dataset (list): A list containing dataset-related information.
        model (object): The federated learning model.
        model_trainer (object, optional): The model trainer object. Defaults to None.
        preprocessed_sampling_lists (list, optional): Preprocessed sampling lists for clients. Defaults to None.

    Returns:
        None
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
    """
    Initialize the server for asynchronous federated learning.

    Args:
        args (object): An object containing the configuration parameters.
        device (object): The device to run the training on (e.g., GPU).
        comm (object): The communication object.
        rank (int): The rank of the current process.
        size (int): The total number of processes.
        model (object): The federated learning model.
        train_data_num (int): The number of training data samples.
        train_data_global (object): The global training dataset.
        test_data_global (object): The global test dataset.
        train_data_local_dict (dict): A dictionary containing local training data for clients.
        test_data_local_dict (dict): A dictionary containing local test data for clients.
        train_data_local_num_dict (dict): A dictionary containing the number of local training data samples for clients.
        model_trainer (object): The model trainer object.
        preprocessed_sampling_lists (list, optional): Preprocessed sampling lists for clients. Defaults to None.

    Returns:
        None
    """
    if model_trainer is None:
        model_trainer = create_model_trainer(model, args)
    model_trainer.set_id(-1)

    # aggregator
    worker_num = size - 1
    aggregator = AsyncFedAVGAggregator(
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
        server_manager = AsyncFedAVGServerManager(args, aggregator, comm, rank, size, backend)
    else:
        server_manager = AsyncFedAVGServerManager(
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
    """
    Initialize a client for asynchronous federated learning.

    Args:
        args (object): An object containing the configuration parameters.
        device (object): The device to run the training on (e.g., GPU).
        comm (object): The communication object.
        process_id (int): The unique ID of the current process.
        size (int): The total number of processes.
        model (object): The federated learning model.
        train_data_num (int): The number of training data samples.
        train_data_local_num_dict (dict): A dictionary containing the number of local training data samples for clients.
        train_data_local_dict (dict): A dictionary containing local training data for clients.
        test_data_local_dict (dict): A dictionary containing local test data for clients.
        model_trainer (object, optional): The model trainer object. Defaults to None.

    Returns:
        None
    """
    client_index = process_id - 1
    if model_trainer is None:
        model_trainer = create_model_trainer(model, args)
    model_trainer.set_id(client_index)
    backend = args.backend
    trainer = AsyncFedAVGTrainer(
        client_index,
        train_data_local_dict,
        train_data_local_num_dict,
        test_data_local_dict,
        train_data_num,
        device,
        args,
        model_trainer,
    )
    client_manager = AsyncFedAVGClientManager(args, trainer, comm, process_id, size, backend)
    client_manager.run()
