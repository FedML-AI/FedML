from mpi4py import MPI

from .GKTClientManager import GKTClientMananger
from .GKTClientTrainer import GKTClientTrainer
from .GKTServerManager import GKTServerMananger
from .GKTServerTrainer import GKTServerTrainer


def FedML_init():
    """
    Initialize the Federated Learning environment.

    Returns:
        comm: The MPI communication object.
        process_id: The ID of the current process.
        worker_number: The total number of worker processes.
    """
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number


def FedML_FedGKT_distributed(
        process_id,
        worker_number,
        device,
        comm,
        model,
        dataset,
        args,
):
    """
    Perform Federated Knowledge Transfer (FedGKT) in a distributed setting.

    Args:
        process_id: The ID of the current process.
        worker_number: The total number of worker processes.
        device: The device (e.g., CPU or GPU) for training.
        comm: The MPI communication object.
        model: A tuple containing client and server models.
        dataset: A list containing dataset-related information.
        args: Additional arguments and settings.

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
    client_model, server_model = model
    if process_id == 0:
        init_server(args, device, comm, process_id, worker_number, server_model)
    else:
        init_client(
            args,
            device,
            comm,
            process_id,
            worker_number,
            client_model,
            train_data_local_dict,
            test_data_local_dict,
            train_data_local_num_dict,
        )


def init_server(args, device, comm, rank, size, model):
    """
    Initialize the Federated Knowledge Transfer (FedGKT) server.

    Args:
        args: Additional arguments and settings.
        device: The device (e.g., CPU or GPU) for training.
        comm: The MPI communication object.
        rank: The rank of the current process.
        size: The total number of processes.
        model: The server model for FedGKT.

    Returns:
        None
    """
    # aggregator
    client_num = size - 1
    server_trainer = GKTServerTrainer(client_num, device, model, args)

    # start the distributed training
    server_manager = GKTServerManager(args, server_trainer, comm, rank, size)
    server_manager.run()


def init_client(
        args,
        device,
        comm,
        process_id,
        size,
        model,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict,
):
    """
    Initialize a FedGKT client.

    Args:
        args: Additional arguments and settings.
        device: The device (e.g., CPU or GPU) for training.
        comm: The MPI communication object.
        process_id: The ID of the current process.
        size: The total number of processes.
        model: The client model for FedGKT.
        train_data_local_dict: A dictionary of local training data.
        test_data_local_dict: A dictionary of local testing data.
        train_data_local_num_dict: A dictionary of the number of local training samples.

    Returns:
        None
    """
    client_ID = process_id - 1

    # 2. initialize the trainer
    trainer = GKTClientTrainer(
        client_ID,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict,
        device,
        model,
        args,
    )

    # 3. start the distributed training
    client_manager = GKTClientManager(args, trainer, comm, process_id, size)
    client_manager.run()
