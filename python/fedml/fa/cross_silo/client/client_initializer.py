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
    return TrainerDistAdapter(
        args,
        client_rank,
        train_data_num,
        train_data_local_num_dict,
        train_data_local_dict,
        local_analyzer,
    )


def get_client_manager_master(args, trainer_dist_adapter, comm, client_rank, client_num, backend):
    return ClientMasterManager(args, trainer_dist_adapter, comm, client_rank, client_num, backend)


def get_client_manager_salve(args, trainer_dist_adapter):
    from .fedml_client_slave_manager import ClientSlaveManager

    return ClientSlaveManager(args, trainer_dist_adapter)
