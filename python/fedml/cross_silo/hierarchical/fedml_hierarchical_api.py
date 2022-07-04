from .fedml_client_master_manager import ClientMasterManager
from .fedml_client_slave_manager import ClientSlaveManager
# from .aggregator_dist_adapter import AggregatorDistAdapter
from .fedml_trainer_dist_adapter import TrainerDistAdapter
from .fedml_server_manager import FedMLServerManager

from .fedml_aggregator import FedMLAggregator
from .utils import get_model_trainer
import logging





# TODO: remove commented code

def FedML_Hierarchical(
    args,
    client_rank,
    client_num,
    comm,
    device,
    dataset,
    model,
    model_trainer=None,
    preprocessed_sampling_lists=None,
):
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

    # process_device = corss_silo_mapping_processes_to_gpu_device_from_yaml_file(
    #     args.rank_in_node, args.proc_rank_in_silo, args.n_proc_in_silo, args.client_num_num, args.silo_gpu_mapping_file
    # )

    # if args.n_proc_in_silo == 0:
    #     assert silo_server_device == process_device, "GPU index mismatch between gpu_mapping and silo_gpu_mapping files"


    if client_rank == 0:
        init_server(
            args,
            device,
            comm,
            client_rank,
            client_num,
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
            client_rank,
            client_num,
            model,
            train_data_num,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            model_trainer,
        )





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


# def get_dist_aggregator(
#     args,
#     device,
#     client_num,
#     model,
#     train_data_num,
#     train_data_global,
#     test_data_global,
#     train_data_local_dict,
#     test_data_local_dict,
#     train_data_local_num_dict,
#     model_trainer,
# ):
#     return AggregatorDistAdapter(
#         args,
#         device,
#         client_num,
#         model,
#         train_data_num,
#         train_data_global,
#         test_data_global,
#         train_data_local_dict,
#         test_data_local_dict,
#         train_data_local_num_dict,
#         model_trainer,
#     )


def get_server_manager(
    args,
    dist_aggregator,
    comm,
    rank,
    client_num,
    backend,
    is_preprocessed=False,
    preprocessed_client_lists=None,
):
    return FedMLServerManager(
        args,
        dist_aggregator,
        comm,
        rank,
        client_num,
        backend,
        is_preprocessed=is_preprocessed,
        preprocessed_client_lists=preprocessed_client_lists,
    )


def get_clinet_manager_master(
    args, trainer_dist_adapter, comm, client_rank, client_num, backend
):
    return ClientMasterManager(
        args, trainer_dist_adapter, comm, client_rank, client_num, backend
    )


def get_clinet_manager_salve(args, trainer_dist_adapter):
    return ClientSlaveManager(args, trainer_dist_adapter)


def init_server(
    args,
    device,
    comm,
    rank,
    client_num,
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

    # start the distributed training
    # backend = args.backend

    # dist_aggregator = get_dist_aggregator(
    #     args,
    #     device,
    #     client_num,
    #     model,
    #     train_data_num,
    #     train_data_global,
    #     test_data_global,
    #     train_data_local_dict,
    #     test_data_local_dict,
    #     train_data_local_num_dict,
    #     model_trainer,
    # )

    # if preprocessed_sampling_lists is None:
    #     server_manager = get_server_manager(
    #         args, dist_aggregator, comm, rank, client_num, backend
    #     )
    # else:
    #     server_manager = get_server_manager(
    #         args,
    #         dist_aggregator,
    #         comm,
    #         rank,
    #         client_num,
    #         backend,
    #         is_preprocessed=True,
    #         preprocessed_client_lists=preprocessed_sampling_lists,
    #     )
    # server_manager.run()

    if model_trainer is None:
        model_trainer = get_model_trainer(model, args)

    model_trainer.set_id(0)

    # aggregator
    aggregator = FedMLAggregator(
        train_data_global,
        test_data_global,
        train_data_num,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict,
        client_num, # Check is client_num same as client_num?
        device,
        args,
        model_trainer,
    )

    # start the distributed training
    backend = args.backend
    if preprocessed_sampling_lists is None:
        server_manager = FedMLServerManager(
            args, aggregator, comm, rank, client_num, # Check is client_num same as client_num?
            backend
        )
    else:
        server_manager = FedMLServerManager(
            args,
            aggregator,
            comm,
            rank,
            client_num, # Check is client_num same as client_num?
            backend,
            is_preprocessed=True,
            preprocessed_client_lists=preprocessed_sampling_lists,
        )
    server_manager.run()


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
    if args.proc_rank_in_silo == 0:
        logging.info("Initiating Client Manager")
        client_manager = get_clinet_manager_master(
            args, trainer_dist_adapter, comm, client_rank, client_num, backend
        )
    else:
        logging.info("Initiating DDP worker")
        client_manager = get_clinet_manager_salve(args, trainer_dist_adapter)
    logging.info("Ruuning Client")
    client_manager.run()
