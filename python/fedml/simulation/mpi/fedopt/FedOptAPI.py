from mpi4py import MPI

from .FedOptAggregator import FedOptAggregator
from .FedOptClientManager import FedOptClientManager
from .FedOptServerManager import FedOptServerManager
from .FedOptTrainer import FedOptTrainer
from ....core import ClientTrainer, ServerAggregator
from ....ml.aggregator.aggregator_creator import create_server_aggregator
from ....ml.trainer.trainer_creator import create_model_trainer


def FedML_init():
    """Initialize the Federated Learning environment using MPI.

    Returns:
        tuple: A tuple containing MPI communication object, process ID, and worker number.
    """
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number


def FedML_FedOpt_distributed(
    args,
    process_id,
    worker_number,
    comm,
    device,
    dataset,
    model,
    client_trainer=None,
    server_aggregator=None,
):
    """Initialize and run the Federated Optimization process.

    Args:
        args: A configuration object containing federated optimization parameters.
        process_id: The process ID.
        worker_number: The total number of workers.
        comm: MPI communication object.
        device: The device (e.g., CPU or GPU) for training.
        dataset: A list containing dataset information.
        model: The machine learning model.
        client_trainer: An optional client trainer object.
        server_aggregator: An optional server aggregator object.

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
    """Initialize the server-side components for federated optimization.

    Args:
        args: A configuration object containing server parameters.
        device: The device (e.g., CPU or GPU) for training.
        comm: MPI communication object.
        rank: The rank of the server process.
        size: The total number of processes.
        model: The machine learning model.
        train_data_num: The number of training data samples.
        train_data_global: Global training data.
        test_data_global: Global test data.
        train_data_local_dict: Dictionary of local training data.
        test_data_local_dict: Dictionary of local test data.
        train_data_local_num_dict: Dictionary of the number of local training data samples.
        server_aggregator: The server aggregator object.

    """
    if server_aggregator is None:
        server_aggregator = create_server_aggregator(model, args)
    server_aggregator.set_id(-1)

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
    """Initialize the client-side components for federated optimization.

    Args:
        args: A configuration object containing client parameters.
        device: The device (e.g., CPU or GPU) for training.
        comm: MPI communication object.
        process_id: The process ID.
        size: The total number of processes.
        model: The machine learning model.
        train_data_num: The number of training data samples.
        train_data_local_num_dict: Dictionary of the number of local training data samples.
        train_data_local_dict: Dictionary of local training data.
        test_data_local_dict: Dictionary of local test data.
        model_trainer: An optional client trainer object.

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
