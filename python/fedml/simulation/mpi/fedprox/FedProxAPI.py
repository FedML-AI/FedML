from mpi4py import MPI

from ....core import ClientTrainer, ServerAggregator
from ....ml.aggregator.aggregator_creator import create_server_aggregator
from ....ml.trainer.trainer_creator import create_model_trainer
from .FedProxAggregator import FedProxAggregator
from .FedProxClientManager import FedProxClientManager
from .FedProxServerManager import FedProxServerManager
from .FedProxTrainer import FedProxTrainer


def FedML_init():
    """
    Initialize the Federated Machine Learning environment.

    Returns:
        tuple: A tuple containing the MPI communication object, process ID, and worker number.
    """
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number


def FedML_FedProx_distributed(
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
    Run the Federated Proximal training process.

    Args:
        args (object): Arguments for configuration.
        process_id (int): The process ID of the current worker.
        worker_number (int): The total number of workers.
        comm (object): Communication object.
        device (object): Device for computation.
        dataset (list): List containing dataset information.
        model (object): Model for training.
        client_trainer (object): Trainer for client-side training (default: None).
        server_aggregator (object): Server aggregator for aggregation (default: None).
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
    Initialize the server for Federated Proximal training.

    Args:
        args (object): Arguments for configuration.
        device (object): Device for computation.
        comm (object): Communication object.
        rank (int): Rank of the server.
        size (int): Total number of participants.
        model (object): Model for training.
        train_data_num (int): Number of training data samples.
        train_data_global (object): Global training data.
        test_data_global (object): Global testing data.
        train_data_local_dict (dict): Dictionary of local training data.
        test_data_local_dict (dict): Dictionary of local testing data.
        train_data_local_num_dict (dict): Dictionary of local training data sizes.
        server_aggregator (object): Server aggregator for aggregation.
    """
    if server_aggregator is None:
        server_aggregator = create_server_aggregator(model, args)
    server_aggregator.set_id(-1)

    # aggregator
    worker_num = size - 1
    aggregator = FedProxAggregator(
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
    server_manager = FedProxServerManager(args, aggregator, comm, rank, size, backend)
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
    Initialize a client for Federated Proximal training.

    Args:
        args (object): Arguments for configuration.
        device (object): Device for computation.
        comm (object): Communication object.
        process_id (int): Process ID of the client.
        size (int): Total number of participants.
        model (object): Model for training.
        train_data_num (int): Number of training data samples.
        train_data_local_num_dict (dict): Dictionary of local training data sizes.
        train_data_local_dict (dict): Dictionary of local training data.
        test_data_local_dict (dict): Dictionary of local testing data.
        model_trainer (object): Trainer for the model (default: None).
    """
    client_index = process_id - 1
    if model_trainer is None:
        model_trainer = create_model_trainer(model, args)
    model_trainer.set_id(client_index)
    backend = args.backend
    trainer = FedProxTrainer(
        client_index,
        train_data_local_dict,
        train_data_local_num_dict,
        test_data_local_dict,
        train_data_num,
        device,
        args,
        model_trainer,
    )
    client_manager = FedProxClientManager(args, trainer, comm, process_id, size, backend)
    client_manager.run()
