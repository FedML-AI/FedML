from mpi4py import MPI

from .FedOptAggregator import FedOptAggregator
from .FedOptClientManager import FedOptClientManager
from .FedOptServerManager import FedOptServerManager
from .FedOptTrainer import FedOptTrainer
from ....core import ClientTrainer, ServerAggregator
from ....ml.aggregator.aggregator_creator import create_server_aggregator
from ....ml.trainer.trainer_creator import create_model_trainer


def FedML_init():
    """
    Initialize the Federated Learning environment.

    Returns:
        tuple: A tuple containing the MPI communicator, process ID, and worker number.
    """
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number


def FedML_FedOptSeq_distributed(
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
    Run the Federated Optimization (FedOpt) distributed training.

    Args:
        args (object): Arguments for configuration.
        process_id (int): Process ID or rank.
        worker_number (int): Total number of workers.
        comm (object): MPI communicator.
        device (object): Device for computation.
        dataset (list): List of dataset elements.
        model (object): Model for training.
        client_trainer (ClientTrainer, optional): Client trainer (default: None).
        server_aggregator (ServerAggregator, optional): Server aggregator (default: None).

    Notes:
        This function orchestrates the FedOpt distributed training process.
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
    Initialize the server for FedOpt distributed training.

    Args:
        args (object): Arguments for configuration.
        device (object): Device for computation.
        comm (object): MPI communicator.
        rank (int): Server's rank.
        size (int): Total number of workers.
        model (object): Model for training.
        train_data_num (int): Number of training data samples.
        train_data_global (object): Global training data.
        test_data_global (object): Global test data.
        train_data_local_dict (dict): Local training data per client.
        test_data_local_dict (dict): Local test data per client.
        train_data_local_num_dict (dict): Number of local training data per client.
        server_aggregator (ServerAggregator, optional): Server aggregator (default: None).

    Notes:
        This function initializes the server and starts distributed training.
    """
    if server_aggregator is None:
        server_aggregator = create_server_aggregator(model, args)
    server_aggregator.set_id(-1)
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
        server_aggregator,
    )

    # start the distributed training
    server_manager = FedOptServerManager(args, aggregator, comm, rank, size)
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
    Initialize a client for FedOpt distributed training.

    Args:
        args (object): Arguments for configuration.
        device (object): Device for computation.
        comm (object): MPI communicator.
        process_id (int): Client's process ID.
        size (int): Total number of workers.
        model (object): Model for training.
        train_data_num (int): Number of training data samples.
        train_data_local_num_dict (dict): Number of local training data per client.
        train_data_local_dict (dict): Local training data per client.
        test_data_local_dict (dict): Local test data per client.
        model_trainer (object, optional): Model trainer (default: None).

    Notes:
        This function initializes a client and runs the training process.
    """
    client_index = process_id - 1
    if model_trainer is None:
        model_trainer = create_model_trainer(model, args)
    model_trainer.set_id(client_index)

    trainer = FedOptTrainer(
        client_index, train_data_local_dict, train_data_local_num_dict, train_data_num, device, args, model_trainer,
    )
    client_manager = FedOptClientManager(args, trainer, comm, process_id, size)
    client_manager.run()
