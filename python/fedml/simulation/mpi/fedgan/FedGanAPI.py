from mpi4py import MPI

from .FedGANAggregator import FedGANAggregator
from .FedGANTrainer import FedGANTrainer
from .FedGanClientManager import FedGANClientManager
from .FedGanServerManager import FedGANServerManager
from .gan_trainer import GANTrainer


def FedML_init():
    """
    Initialize the MPI communication and return necessary information.

    Returns:
        tuple: A tuple containing the MPI communication object, process ID, and worker number.
    """
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()

    return comm, process_id, worker_number



def FedML_FedGan_distributed(
        args,
        process_id,
        worker_number,
        device,
        comm,
        model,
        dataset,
        model_trainer=None,
        preprocessed_sampling_lists=None,
):
    """
    Initialize and run the Federated GAN distributed training.

    Args:
        args: Configuration arguments.
        process_id (int): The process ID of the current worker.
        worker_number (int): Total number of workers.
        device: Torch device for computation (e.g., 'cuda' or 'cpu').
        comm: MPI communication object.
        model: GAN model to be trained.
        dataset: Dataset information including training and testing data.
        model_trainer: Model trainer object for training and testing.
        preprocessed_sampling_lists: Preprocessed client sampling lists.

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

    netg, netd = model

    if model_trainer is None:
        model_trainer = GANTrainer(netd, netg)

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
    Initialize the server for Federated GAN training.

    Args:
        args: Configuration arguments.
        device: Torch device for computation (e.g., 'cuda' or 'cpu').
        comm: MPI communication object.
        rank (int): Rank of the current process.
        size (int): Total number of processes.
        model: GAN model to be trained.
        train_data_num: Total number of training samples.
        train_data_global: Global training dataset.
        test_data_global: Global testing dataset.
        train_data_local_dict: Dictionary of local training datasets for each worker.
        test_data_local_dict: Dictionary of local testing datasets for each worker.
        train_data_local_num_dict: Dictionary of the number of local training samples for each worker.
        model_trainer: Model trainer object for training and testing.
        preprocessed_sampling_lists: Preprocessed client sampling lists.

    """
    if model_trainer is None:
        pass

    model_trainer.set_id(-1)

    # aggregator
    worker_num = size - 1

    aggregator = FedGANAggregator(
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
        server_manager = FedGANServerManager(
            args, aggregator, comm, rank, size, backend
        )
    else:
        server_manager = FedGANServerManager(
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
    print("server init done")


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
    Initialize a client for Federated GAN training.

    Args:
        args: Configuration arguments.
        device: Torch device for computation (e.g., 'cuda' or 'cpu').
        comm: MPI communication object.
        process_id (int): The process ID of the current client.
        size (int): Total number of processes.
        model: GAN model to be trained.
        train_data_num: Total number of training samples.
        train_data_local_num_dict: Dictionary of the number of local training samples for each worker.
        train_data_local_dict: Dictionary of local training datasets for each worker.
        test_data_local_dict: Dictionary of local testing datasets for each worker.
        model_trainer: Model trainer object for training and testing.

    """
    client_index = process_id - 1

    model_trainer.set_id(client_index)
    backend = args.backend
    trainer = FedGANTrainer(
        client_index,
        train_data_local_dict,
        train_data_local_num_dict,
        test_data_local_dict,
        train_data_num,
        device,
        args,
        model_trainer,
    )
    client_manager = FedGANClientManager(args, trainer, comm, process_id, size, backend)
    client_manager.run()
