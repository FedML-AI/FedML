from fedml.constants import (
    FEDML_CROSS_SILO_CUSTOMIZED_HIERARCHICAL_KEY,
    FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL,
    FEDML_CROSS_SILO_SCENARIO_HORIZONTAL,
)
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
        args: The command-line arguments.
        device: The device to perform computations on.
        comm: The communication backend.
        client_rank: The rank of the client.
        client_num: The total number of clients.
        model: The federated learning model.
        train_data_num: The total number of training data samples.
        train_data_local_num_dict: A dictionary mapping client IDs to the number of local training data samples.
        train_data_local_dict: A dictionary mapping client IDs to their local training data.
        test_data_local_dict: A dictionary mapping client IDs to their local testing data.
        model_trainer: The model trainer (optional).

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
    if (
            args.scenario == FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL or
            (
                args.scenario == FEDML_CROSS_SILO_SCENARIO_HORIZONTAL and
                getattr(args, FEDML_CROSS_SILO_CUSTOMIZED_HIERARCHICAL_KEY, False)
            )
    ):
        if args.proc_rank_in_silo == 0:
            client_manager = get_client_manager_master(
                args, trainer_dist_adapter, comm, client_rank, client_num, backend
            )

        else:
            client_manager = get_client_manager_salve(
                args, trainer_dist_adapter)

    elif args.scenario == FEDML_CROSS_SILO_SCENARIO_HORIZONTAL:
        client_manager = get_client_manager_master(
            args, trainer_dist_adapter, comm, client_rank, client_num, backend)

    else:
        raise RuntimeError(
            "we do not support {}. Please check whether this is typo.".format(args.scenario))

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
    Get a trainer distributed adapter.

    Args:
        args: The command-line arguments.
        device: The device to perform computations on.
        client_rank: The rank of the client.
        model: The federated learning model.
        train_data_num: The total number of training data samples.
        train_data_local_num_dict: A dictionary mapping client IDs to the number of local training data samples.
        train_data_local_dict: A dictionary mapping client IDs to their local training data.
        test_data_local_dict: A dictionary mapping client IDs to their local testing data.
        model_trainer: The model trainer (optional).

    Returns:
        TrainerDistAdapter: The trainer distributed adapter.
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
    Get the federated learning client manager for the master.

    Args:
        args: The command-line arguments.
        trainer_dist_adapter: The trainer distributed adapter.
        comm: The communication backend.
        client_rank: The rank of the client.
        client_num: The total number of clients.
        backend: The communication backend.

    Returns:
        ClientMasterManager: The federated learning client manager for the master.
    """
    return ClientMasterManager(args, trainer_dist_adapter, comm, client_rank, client_num, backend)


def get_client_manager_salve(args, trainer_dist_adapter):
    """
    Get the federated learning client manager for a slave.

    Args:
        args: The command-line arguments.
        trainer_dist_adapter: The trainer distributed adapter.

    Returns:
        ClientSlaveManager: The federated learning client manager for a slave.
    """
    from .fedml_client_slave_manager import ClientSlaveManager

    return ClientSlaveManager(args, trainer_dist_adapter)
