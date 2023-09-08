from .FedAVGAggregator import FedAVGAggregator
from .FedAVGTrainer import FedAVGTrainer
from .FedAvgClientManager import FedAVGClientManager
from .FedAvgServerManager import FedAVGServerManager
from ....core.dp.fedml_differential_privacy import FedMLDifferentialPrivacy
from ....core.security.fedml_attacker import FedMLAttacker
from ....core.security.fedml_defender import FedMLDefender
from ....ml.aggregator.aggregator_creator import create_server_aggregator
from ....ml.trainer.trainer_creator import create_model_trainer


def FedML_FedAvgSeq_distributed(
    args, process_id, worker_number, comm, device, dataset, model, client_trainer=None, server_aggregator=None
):
    """
    Function to initialize and run federated learning in a distributed environment using the FedAvg algorithm.

    Args:
        args (Namespace): Command-line arguments and configuration.
        process_id (int): The unique identifier for the current process.
        worker_number (int): The total number of worker processes.
        comm (object): The communication backend (e.g., MPI).
        device (str): The device (e.g., "cpu" or "cuda") for training.
        dataset (list): List containing dataset information.
        model (nn.Module): The federated learning model.
        client_trainer (object, optional): An instance of the client model trainer. Defaults to None.
        server_aggregator (object, optional): An instance of the server aggregator. Defaults to None.
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
            server_aggregator,
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

# Rest of the code...

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
    server_aggregator,
):
    """
    Initialize and run the federated learning server.

    Args:
        args (Namespace): Command-line arguments and configuration.
        device (str): The device (e.g., "cpu" or "cuda") for training.
        comm (object): The communication backend (e.g., MPI).
        rank (int): The rank of the server process.
        size (int): The total number of processes.
        model (nn.Module): The federated learning model.
        train_data_num (int): The total number of training samples.
        train_data_global (Dataset): The global training dataset.
        test_data_global (Dataset): The global test dataset.
        train_data_local_dict (dict): A dictionary of local training datasets.
        test_data_local_dict (dict): A dictionary of local test datasets.
        train_data_local_num_dict (dict): A dictionary of the number of samples in each local training dataset.
        server_aggregator (object): An instance of the server aggregator.
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
    client_trainer=None,
):
    """
    Initialize and run a federated learning client.

    Args:
        args (Namespace): Command-line arguments and configuration.
        device (str): The device (e.g., "cpu" or "cuda") for training.
        comm (object): The communication backend (e.g., MPI).
        process_id (int): The unique identifier for the client process.
        size (int): The total number of processes.
        model (nn.Module): The federated learning model.
        train_data_num (int): The total number of training samples.
        train_data_local_num_dict (dict): A dictionary of the number of samples in each local training dataset.
        train_data_local_dict (dict): A dictionary of local training datasets.
        test_data_local_dict (dict): A dictionary of local test datasets.
        client_trainer (object, optional): An instance of the client model trainer. Defaults to None.
    """
    client_index = process_id - 1
    if client_trainer is None:
        client_trainer = create_model_trainer(model, args)
    client_trainer.set_id(client_index)
    backend = args.backend
    trainer = FedAVGTrainer(
        client_index,
        train_data_local_dict,
        train_data_local_num_dict,
        test_data_local_dict,
        train_data_num,
        device,
        args,
        client_trainer,
    )
    client_manager = FedAVGClientManager(args, trainer, comm, process_id, size, backend)
    client_manager.run()
