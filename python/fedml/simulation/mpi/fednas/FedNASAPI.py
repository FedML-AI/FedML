from mpi4py import MPI

from fedml.core import ClientTrainer, ServerAggregator
from .FedNASAggregator import FedNASAggregator
from .FedNASClientManager import FedNASClientManager
from .FedNASServerManager import FedNASServerManager
from .FedNASTrainer import FedNASTrainer


def FedML_init():
    """
    Initialize the Federated Machine Learning environment using MPI (Message Passing Interface).

    Returns:
        Tuple: A tuple containing the MPI communicator (`comm`), process ID (`process_id`), and worker number (`worker_number`).
    """
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number

def FedML_FedNAS_distributed(
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
    Initialize and run the Federated NAS (Neural Architecture Search) distributed training process.

    Args:
        args: Command-line arguments and configurations.
        process_id (int): The process ID of the current worker.
        worker_number (int): The total number of workers.
        comm: The MPI communicator.
        device: The device (e.g., GPU) to run the training on.
        dataset: A list containing dataset information.
        model: The neural network model.
        client_trainer (ClientTrainer, optional): The client trainer instance.
        server_aggregator (ServerAggregator, optional): The server aggregator instance.
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
            args, device, comm, process_id, worker_number, model, train_data_num, train_data_global, test_data_global,
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
            test_data_num,
            train_data_local_dict,
            test_data_local_dict,
        )

def init_server(
    args, device, comm, process_id, worker_number, model, train_data_num, train_data_global, test_data_global,
):
    """
    Initialize and run the server component of the Federated NAS distributed training.

    Args:
        args: Command-line arguments and configurations.
        device: The device (e.g., GPU) to run the training on.
        comm: The MPI communicator.
        process_id (int): The process ID of the current worker.
        worker_number (int): The total number of workers.
        model: The neural network model.
        train_data_num: The number of training data samples.
        train_data_global: The global training data.
        test_data_global: The global testing data.
    """
    # aggregator
    client_num = worker_number - 1
    aggregator = FedNASAggregator(train_data_global, test_data_global, train_data_num, client_num, model, device, args,)

    # start the distributed training
    server_manager = FedNASServerManager(args, comm, process_id, worker_number, aggregator)
    server_manager.run()

def init_client(
    args,
    device,
    comm,
    process_id,
    worker_number,
    model,
    train_data_num,
    local_data_num,
    train_data_local,
    test_data_local,
):
    """
    Initialize and run the client component of the Federated NAS distributed training.

    Args:
        args: Command-line arguments and configurations.
        device: The device (e.g., GPU) to run the training on.
        comm: The MPI communicator.
        process_id (int): The process ID of the current worker.
        worker_number (int): The total number of workers.
        model: The neural network model.
        train_data_num: The number of training data samples.
        local_data_num: The number of local training data samples.
        train_data_local: The local training data.
        test_data_local: The local testing data.
    """
    # trainer
    client_ID = process_id - 1
    trainer = FedNASTrainer(
        client_ID, train_data_local, test_data_local, local_data_num, train_data_num, model, device, args,
    )

    client_manager = FedNASClientManager(args, comm, process_id, worker_number, trainer)
    client_manager.run()
