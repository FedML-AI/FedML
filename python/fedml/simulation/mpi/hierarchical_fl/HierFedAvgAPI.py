from .HierFedAvgCloudAggregator import HierFedAVGCloudAggregator
from .HierFedAvgCloudManager import HierFedAVGCloudManager
from .HierFedAvgEdgeManager import HierFedAVGEdgeManager
from .HierGroup import HierGroup
from .utils import analyze_clients_type, hetero_partition_groups, stats_group
from ....core import ClientTrainer, ServerAggregator
from ....core.dp.fedml_differential_privacy import FedMLDifferentialPrivacy
from ....core.security.fedml_attacker import FedMLAttacker
from ....core.security.fedml_defender import FedMLDefender
from ....ml.aggregator.aggregator_creator import create_server_aggregator
from ....ml.trainer.trainer_creator import create_model_trainer
from ....core.distributed.topology.symmetric_topology_manager import SymmetricTopologyManager


import numpy as np
import wandb

def FedML_HierFedAvg_distributed(
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

    FedMLAttacker.get_instance().init(args)
    FedMLDefender.get_instance().init(args)
    FedMLDifferentialPrivacy.get_instance().init(args)

    if process_id == 0:
        init_cloud_server(
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
            server_aggregator,
            class_num
        )
    else:
        init_edge_server_clients(
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
            client_trainer,
            server_aggregator
        )


def init_cloud_server(
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
    server_aggregator,
    class_num
):
    if server_aggregator is None:
        server_aggregator = create_server_aggregator(model, args)
    server_aggregator.set_id(-1)

    worker_num = size - 1

    # set up topology
    topology_manager = None
    if hasattr(args, "topo_name"):
        topology_manager = SymmetricTopologyManager(worker_num, args)
        topology_manager.generate_custom_topology(args)

    # aggregator
    aggregator = HierFedAVGCloudAggregator(
        train_data_global,
        test_data_global,
        train_data_num,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict,
        worker_num,
        device,
        args,
        server_aggregator
    )

    # start the distributed training
    backend = args.backend
    group_indexes, group_to_client_indexes = setup_clients(args, train_data_local_dict, class_num)

    # print group detail
    stats_group(group_to_client_indexes, train_data_local_dict, train_data_local_num_dict, class_num, args)

    server_manager = HierFedAVGCloudManager(args, aggregator, group_indexes, group_to_client_indexes,
                                            comm, rank, size, backend, topology_manager)
    server_manager.send_init_msg()
    server_manager.run()


def init_edge_server_clients(
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
    group,
    model_trainer=None,
):

    if model_trainer is None:
        model_trainer = create_model_trainer(model, args)

    edge_index = process_id - 1
    backend = args.backend

    # Client assignment is decided on cloud server and the information will be communicated later
    group = HierGroup(
        edge_index,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict,
        args,
        device,
        model,
        model_trainer
    )

    edge_manager = HierFedAVGEdgeManager(group, args, comm, process_id, size, backend)
    edge_manager.run()


def setup_clients(
    args,
    train_data_local_dict,
    class_num
    ):

    if args.group_method == "random":
        group_indexes = np.random.randint(
            0, args.group_num, args.client_num_in_total
        )
        group_to_client_indexes = {}
        for client_idx, group_idx in enumerate(group_indexes):
            if not group_idx in group_to_client_indexes:
                group_to_client_indexes[group_idx] = []
            group_to_client_indexes[group_idx].append(client_idx)
    elif args.group_method == "hetero":
        clients_type_list = analyze_clients_type(train_data_local_dict, class_num, num_type=args.group_num)
        group_indexes, group_to_client_indexes = hetero_partition_groups(clients_type_list,
                                                                          args.group_num,
                                                                          alpha=args.group_alpha)

    return group_indexes, group_to_client_indexes