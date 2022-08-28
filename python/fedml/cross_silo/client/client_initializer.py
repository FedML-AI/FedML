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
    return ClientMasterManager(args, trainer_dist_adapter, comm, client_rank, client_num, backend)


def get_client_manager_salve(args, trainer_dist_adapter):
    from .fedml_client_slave_manager import ClientSlaveManager

    return ClientSlaveManager(args, trainer_dist_adapter)
