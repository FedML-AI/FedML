import logging

from mpi4py import MPI

from .FedSegAggregator import FedSegAggregator
from .FedSegClientManager import FedSegClientManager
from .FedSegServerManager import FedSegServerManager
from .FedSegTrainer import FedSegTrainer
from .MyModelTrainer import MyModelTrainer


def FedML_init():
    """
    Initialize the federated learning environment.

    Returns:
        tuple: A tuple containing the MPI communicator (`comm`), process ID (`process_id`), and worker number (`worker_number`).
    """
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
    """
    Initialize and run the federated Segmentation training process.

    Args:
        process_id (int): The ID of the current process.
        worker_number (int): The total number of workers (including the server).
        device: The device on which the model is trained.
        comm: The MPI communicator.
        model: The neural network model.
        train_data_num: The number of training data samples.
        train_data_local_num_dict: A dictionary containing the number of local training data samples for each worker.
        train_data_local_dict: A dictionary containing the local training data for each worker.
        test_data_local_dict: A dictionary containing the local testing data for each worker.
        args: Additional arguments for the federated learning setup.
        model_trainer: The model trainer for training the model (optional).

    Notes:
        - If `process_id` is 0, it initializes the server. Otherwise, it initializes a client.
    """

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
    """
    Initialize the federated learning server.

    Args:
        args: Additional arguments for the server initialization.
        device: The device on which the model is trained.
        comm: The MPI communicator.
        rank (int): The rank of the current process.
        size (int): The total number of processes.
        model: The neural network model.
        model_trainer: The model trainer for training the model (optional).

    Notes:
        This function initializes the server for federated Segmentation training.
    """
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
    """
    Initialize and run a federated learning client.

    Args:
        args: Additional arguments for the client initialization.
        device: The device on which the model is trained.
        comm: The MPI communicator.
        process_id (int): The ID of the current client process.
        size (int): The total number of processes.
        model: The neural network model.
        train_data_num: The number of training data samples.
        train_data_local_num_dict: A dictionary containing the number of local training data samples for each client.
        train_data_local_dict: A dictionary containing the local training data for each client.
        test_data_local_dict: A dictionary containing the local testing data for each client.
        model_trainer: The model trainer for training the model (optional).

    Notes:
        This function initializes and runs a federated learning client.
    """

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
