from fedml.constants import FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL, FEDML_CROSS_SILO_SCENARIO_HORIZONTAL
from .fedml_client_master_manager import ClientMasterManager
from .fedml_trainer_dist_adapter import TrainerDistAdapter


def init_client(
    args,
    device,
    comm,
    client_rank,
    client_num,
    model,
    train_data_num,
    train_data_local_num_dict,
    train_data_local_dict,
    test_data_local_dict,
    model_trainer=None,
):
    """
    Initialize and run a federated learning client.

    Args:
        args: Arguments and configuration for the client.
        device: The device on which the client should run (e.g., 'cpu' or 'cuda').
        comm: The communication backend for distributed training.
        client_rank: The rank or identifier of this client.
        client_num: The total number of clients in the federated learning scenario.
        model: The machine learning model to be trained.
        train_data_num: The number of training data points.
        train_data_local_num_dict: A dictionary mapping client IDs to the number of local training data points.
        train_data_local_dict: A dictionary mapping client IDs to their local training data.
        test_data_local_dict: A dictionary mapping client IDs to their local testing data.
        model_trainer: An optional custom model trainer.

    Returns:
        None
    """
    backend = args.backend

    trainer_dist_adapter = get_trainer_dist_adapter(
        args,
        device,
        client_rank,
        model,
        train_data_num,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        model_trainer,
    )
    if args.scenario == FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL:
        if args.proc_rank_in_silo == 0:

            client_manager = get_client_manager_master(
                args, trainer_dist_adapter, comm, client_rank, client_num, backend
            )

        else:
            client_manager = get_client_manager_salve(args, trainer_dist_adapter)

    elif args.scenario == FEDML_CROSS_SILO_SCENARIO_HORIZONTAL:

        client_manager = get_client_manager_master(args, trainer_dist_adapter, comm, client_rank, client_num, backend)

    else:
        raise Exception("we do not support {}. Please check whether this is typo.".format(args.scenario))

    client_manager.run()


def get_trainer_dist_adapter(
    args,
    device,
    client_rank,
    model,
    train_data_num,
    train_data_local_num_dict,
    train_data_local_dict,
    test_data_local_dict,
    model_trainer,
):
    """
    Get a distributed trainer adapter for the federated learning client.

    Args:
        args: Arguments and configuration for the client.
        device: The device on which the client should run (e.g., 'cpu' or 'cuda').
        client_rank: The rank or identifier of this client.
        model: The machine learning model to be trained.
        train_data_num: The number of training data points.
        train_data_local_num_dict: A dictionary mapping client IDs to the number of local training data points.
        train_data_local_dict: A dictionary mapping client IDs to their local training data.
        test_data_local_dict: A dictionary mapping client IDs to their local testing data.
        model_trainer: An optional custom model trainer.

    Returns:
        TrainerDistAdapter: A distributed trainer adapter.
    """
    return TrainerDistAdapter(
        args,
        device,
        client_rank,
        model,
        train_data_num,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        model_trainer,
    )


def get_client_manager_master(args, trainer_dist_adapter, comm, client_rank, client_num, backend):
    """
    Get a federated learning client manager for the master client in the hierarchical scenario.

    Args:
        args: Arguments and configuration for the client.
        trainer_dist_adapter: A distributed trainer adapter.
        comm: The communication backend for distributed training.
        client_rank: The rank or identifier of this client.
        client_num: The total number of clients in the federated learning scenario.
        backend: The backend for distributed training (e.g., 'nccl' or 'gloo').

    Returns:
        ClientMasterManager: A federated learning client manager for the master client.
    """
    return ClientMasterManager(args, trainer_dist_adapter, comm, client_rank, client_num, backend)


def get_client_manager_salve(args, trainer_dist_adapter):
    """
    Get a federated learning client manager for a slave client in the hierarchical scenario.

    Args:
        args: Arguments and configuration for the client.
        trainer_dist_adapter: A distributed trainer adapter.

    Returns:
        ClientSlaveManager: A federated learning client manager for a slave client.
    """
    from .fedml_client_slave_manager import ClientSlaveManager

    return ClientSlaveManager(args, trainer_dist_adapter)
