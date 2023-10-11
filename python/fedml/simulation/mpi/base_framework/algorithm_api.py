from mpi4py import MPI

from .central_manager import BaseCentralManager
from .central_worker import BaseCentralWorker
from .client_manager import BaseClientManager
from .client_worker import BaseClientWorker


def FedML_init():
    """
    Initialize the MPI communication and retrieve process information.

    Returns:
        comm (object): MPI communication object.
        process_id (int): Unique ID of the current process.
        worker_number (int): Total number of worker processes.
    """
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number


def FedML_Base_distributed(args, process_id, worker_number, comm):
    """
    Run the base distributed federated learning process.

    Args:
        args (object): An object containing the configuration parameters.
        process_id (int): Unique ID of the current process.
        worker_number (int): Total number of worker processes.
        comm (object): MPI communication object.

    Returns:
        None
    """
    if process_id == 0:
        init_central_worker(args, comm, process_id, worker_number)
    else:
        init_client_worker(args, comm, process_id, worker_number)


def init_central_worker(args, comm, process_id, size):
    """
    Initialize the central worker for distributed federated learning.

    Args:
        args (object): An object containing the configuration parameters.
        comm (object): MPI communication object.
        process_id (int): Unique ID of the current process.
        size (int): Total number of processes.

    Returns:
        None
    """
    # aggregator
    client_num = size - 1
    aggregator = BaseCentralWorker(client_num, args)

    # start the distributed training
    server_manager = BaseCentralManager(args, comm, process_id, size, aggregator)
    server_manager.run()


def init_client_worker(args, comm, process_id, size):
    """
    Initialize a client worker for distributed federated learning.

    Args:
        args (object): An object containing the configuration parameters.
        comm (object): MPI communication object.
        process_id (int): Unique ID of the current process.
        size (int): Total number of processes.

    Returns:
        None
    """
    # trainer
    client_ID = process_id - 1
    trainer = BaseClientWorker(client_ID)

    client_manager = BaseClientManager(args, comm, process_id, size, trainer)
    client_manager.run()
