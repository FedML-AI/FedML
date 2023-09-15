from fedml.constants import FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL, FEDML_CROSS_SILO_SCENARIO_HORIZONTAL
from .fedml_client_master_manager import ClientMasterManager
from .fedml_trainer_dist_adapter import TrainerDistAdapter


def init_client(
    args,
    comm,
    client_rank,
    client_num,
    train_data_num,
    train_data_local_num_dict,
    train_data_local_dict,
    local_analyzer=None,
):
    """
    Initialize the federated learning client.

    Args:
        args: Configuration arguments.
        comm: Communication object.
        client_rank (int): The rank of the client.
        client_num (int): The total number of clients.
        train_data_num (int): The total number of training data samples.
        train_data_local_num_dict (dict): A dictionary mapping client indices to the number of local training samples.
        train_data_local_dict (dict): A dictionary mapping client indices to their local training data.
        local_analyzer: Local analyzer for the client (optional).

    Returns:
        None
    """
    backend = args.backend

    trainer_dist_adapter = get_trainer_dist_adapter(
        args,
        client_rank,
        train_data_num,
        train_data_local_num_dict,
        train_data_local_dict,
        local_analyzer,
    )

    # horizontal
    client_manager = get_client_manager_master(args, trainer_dist_adapter, comm, client_rank, client_num, backend)

    client_manager.run()


def get_trainer_dist_adapter(
    args,
    client_rank,
    train_data_num,
    train_data_local_num_dict,
    train_data_local_dict,
    local_analyzer,
):
    """
    Get the trainer distribution adapter.

    Args:
        args: Configuration arguments.
        client_rank (int): The rank of the client.
        train_data_num (int): The total number of training data samples.
        train_data_local_num_dict (dict): A dictionary mapping client indices to the number of local training samples.
        train_data_local_dict (dict): A dictionary mapping client indices to their local training data.
        local_analyzer: Local analyzer for the client.

    Returns:
        TrainerDistAdapter: The trainer distribution adapter.
    """
    return TrainerDistAdapter(
        args,
        client_rank,
        train_data_num,
        train_data_local_num_dict,
        train_data_local_dict,
        local_analyzer,
    )


def get_client_manager_master(args, trainer_dist_adapter, comm, client_rank, client_num, backend):
    """
    Get the client master manager.

    Args:
        args: Configuration arguments.
        trainer_dist_adapter: Trainer distribution adapter.
        comm: Communication object.
        client_rank (int): The rank of the client.
        client_num (int): The total number of clients.
        backend: Backend for distributed training.

    Returns:
        ClientMasterManager: The client master manager.
    """
    return ClientMasterManager(args, trainer_dist_adapter, comm, client_rank, client_num, backend)


def get_client_manager_salve(args, trainer_dist_adapter):
    """
    Get the client slave manager.

    Args:
        args: Configuration arguments.
        trainer_dist_adapter: Trainer distribution adapter.

    Returns:
        ClientSlaveManager: The client slave manager.
    """
    from .fedml_client_slave_manager import ClientSlaveManager

    return ClientSlaveManager(args, trainer_dist_adapter)
